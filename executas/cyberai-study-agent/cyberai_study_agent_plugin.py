"""Anna Executa for cyberai-study-agent.

Port of the original RAG pipeline (github.com/DavideDeplano/cyberai-study-agent)
to Anna AI OS.

Split of responsibilities:

* This plugin (runs on the user's Anna Agent) handles everything local:
  PDF text extraction, chunking, the vector index on disk, similarity
  search and prompt assembly.
* The app UI calls the Anna host directly for embeddings
  (``anna.llm.embed``) and answers (``anna.llm.complete``). In local dev the
  plugin-side reverse-RPC path never received a reply, while the direct
  iframe path works, so model calls live in the UI.

Tools:
    extract_pdf(path, course?)                  -> chunks to embed
    extract_pdf_bytes(name, data, course?)      -> same, from a base64 upload
    add_chunks(source, course, chunks, replace) -> store embedded chunks
    search(vector, question, top_k?, course?, history?) -> prompt + sources
    stats()                                     -> indexed material summary
    remove_source(source)                       -> drop one PDF
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

# ─── Configuration ───────────────────────────────────────────────────

# Index location: one JSON file in the user's home directory. The
# Executa runs on the user's own Anna Agent, so this stays on their
# machine.
DATA_DIR = Path.home() / ".cyberai-study-agent-anna"
INDEX_PATH = DATA_DIR / "index.json"

# Word-based chunking. 350 words is roughly 450-500 tokens, close to the
# token budget used by the original project.
CHUNK_WORDS = 350
OVERLAP_WORDS = 50

ANSWER_SYSTEM_INSTRUCTION = (
    "You are a study assistant for a Master's student in Cybersecurity and "
    "AI. Answer strictly based on the provided context excerpts from the "
    "student's study materials. Every factual claim MUST cite its source "
    "using the format [source, p.PAGE]. If the context does not contain "
    "enough information to answer, say so explicitly instead of guessing. "
    "Answer in the same language as the user's question. "
    "Do not use LaTeX or $...$ math delimiters: write formulas in plain "
    "text with Unicode symbols (for example X = {x₀, x₁}, δ(x, e))."
)

MANIFEST = {
    "display_name": "CyberAI Study Agent",
    "version": "0.1.0",
    "description": "Indexes study PDFs and retrieves page-cited excerpts for questions.",
    "author": "Davide Deplano",
    "tools": [
        {
            "name": "extract_pdf",
            "description": "Extract and chunk a local PDF. Returns the chunks to embed.",
            "parameters": [
                {"name": "path", "type": "string", "description": "Absolute path to the PDF.", "required": True},
                {"name": "course", "type": "string", "description": "Optional course label.", "required": False},
            ],
        },
        {
            "name": "extract_pdf_bytes",
            "description": "Extract and chunk a PDF sent as base64. Returns the chunks to embed.",
            "parameters": [
                {"name": "name", "type": "string", "description": "File name.", "required": True},
                {"name": "data", "type": "string", "description": "Base64-encoded PDF.", "required": True},
                {"name": "course", "type": "string", "description": "Optional course label.", "required": False},
            ],
        },
        {
            "name": "add_chunks",
            "description": "Store embedded chunks of one PDF in the index.",
            "parameters": [
                {"name": "source", "type": "string", "description": "PDF file name.", "required": True},
                {"name": "course", "type": "string", "description": "Course label.", "required": False},
                {"name": "chunks", "type": "array", "description": "[{page, text, vector}]", "required": True},
                {"name": "replace", "type": "boolean", "description": "Drop existing chunks of this source first.", "required": False},
            ],
        },
        {
            "name": "search",
            "description": "Find the most similar chunks and build the answer prompt.",
            "parameters": [
                {"name": "vector", "type": "array", "description": "Embedding of the question.", "required": True},
                {"name": "question", "type": "string", "description": "The question.", "required": True},
                {"name": "top_k", "type": "integer", "description": "Excerpts to retrieve (default 5).", "required": False},
                {"name": "course", "type": "string", "description": "Restrict to one course.", "required": False},
                {"name": "history", "type": "array", "description": "Previous turns as [{role, text}].", "required": False},
            ],
        },
        {"name": "stats", "description": "Summary of the indexed material.", "parameters": []},
        {
            "name": "remove_source",
            "description": "Remove one PDF from the index.",
            "parameters": [
                {"name": "source", "type": "string", "description": "File name as shown by stats.", "required": True},
            ],
        },
    ],
    "runtime": {"type": "uv", "min_version": "0.1.0"},
}

# ─── Index storage ───────────────────────────────────────────────────

_index_lock = threading.Lock()


def _load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _save_index(chunks: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    tmp.replace(INDEX_PATH)


# ─── Ingestion ───────────────────────────────────────────────────────


def extract_pages(pdf) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page with text, 1-indexed.

    ``pdf`` is a filesystem path or a binary file-like object.
    """
    t0 = time.monotonic()
    reader = PdfReader(pdf if hasattr(pdf, "read") else str(pdf))
    total = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    print(f"extract: {total} pages read in {time.monotonic() - t0:.1f}s", file=sys.stderr, flush=True)
    return pages


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def _chunked(source: str, course: str, pages: list[tuple[int, str]]) -> dict:
    if not pages:
        raise ValueError("no extractable text: this looks like a scanned PDF without a text layer")
    chunks = [{"page": p, "text": piece} for p, text in pages for piece in chunk_text(text)]
    return {"source": source, "course": course or "", "pages": len(pages), "chunks": chunks}


def extract_pdf(path: str, course: str = "") -> dict:
    """Extract a PDF from a local path."""
    pdf_path = Path(path.strip().strip('"')).expanduser()
    if not pdf_path.is_file():
        raise ValueError(f"file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("only .pdf files are supported")
    return _chunked(pdf_path.name, course, extract_pages(pdf_path))


def extract_pdf_bytes(name: str, data: str, course: str = "") -> dict:
    """Extract a PDF sent by the UI file picker as base64."""
    if not name.lower().endswith(".pdf"):
        raise ValueError("only .pdf files are supported")
    raw = base64.b64decode(data)
    if not raw.startswith(b"%PDF"):
        raise ValueError("this file is not a valid PDF")
    return _chunked(Path(name).name, course, extract_pages(io.BytesIO(raw)))


def add_chunks(source: str, chunks: list, course: str = "", replace: bool = False) -> dict:
    if not source:
        raise ValueError("source must be non-empty")
    records = []
    for c in chunks or []:
        vec = c.get("vector")
        if not isinstance(vec, list) or not vec:
            raise ValueError("every chunk needs a non-empty vector")
        records.append({"source": source, "course": course or "", "page": int(c["page"]), "text": c["text"], "vector": vec})
    with _index_lock:
        index = _load_index()
        if replace:
            index = [c for c in index if c["source"] != source]
        index.extend(records)
        _save_index(index)
        total = sum(1 for c in index if c["source"] == source)
    return {"source": source, "added": len(records), "chunks_for_source": total}


# ─── Retrieval and prompt ────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_context(chunks: list[dict]) -> str:
    # Same format as the original project: the header string is the one
    # the system prompt asks the model to cite.
    return "\n\n---\n\n".join(f"Source: {c['source']}, page {c['page']}\n{c['text']}" for c in chunks)


def build_history(history: list[dict]) -> str:
    lines = []
    for turn in history or []:
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('text', '')}")
    return "\n".join(lines)


def search(vector: list, question: str, top_k: int = 5, course: str = "", history: list | None = None) -> dict:
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be non-empty")
    with _index_lock:
        index = _load_index()
    if course:
        index = [c for c in index if c.get("course") == course]
    if not index:
        return {"prompt": None, "system_prompt": ANSWER_SYSTEM_INSTRUCTION, "sources": []}

    top = sorted(index, key=lambda c: _cosine(vector, c["vector"]), reverse=True)[: max(1, int(top_k or 5))]
    context = build_context(top)
    hist = build_history(history or [])
    if hist:
        prompt = (
            f"Conversation so far:\n\n{hist}\n\n---\n\n"
            f"Context excerpts from study materials:\n\n{context}\n\n---\n\n"
            f"Current question: {question}"
        )
    else:
        prompt = f"Context excerpts from study materials:\n\n{context}\n\n---\n\nQuestion: {question}"
    return {
        "prompt": prompt,
        "system_prompt": ANSWER_SYSTEM_INSTRUCTION,
        "sources": [{"source": c["source"], "page": c["page"], "text": c["text"]} for c in top],
    }


# ─── Index management ────────────────────────────────────────────────


def stats() -> dict:
    with _index_lock:
        index = _load_index()
    per_source: dict[str, dict] = {}
    for c in index:
        s = per_source.setdefault(c["source"], {"source": c["source"], "course": c.get("course", ""), "chunks": 0, "pages": set()})
        s["chunks"] += 1
        s["pages"].add(c["page"])
    sources = [{**s, "pages": len(s["pages"])} for s in per_source.values()]
    return {"total_chunks": len(index), "sources": sorted(sources, key=lambda s: s["source"])}


def remove_source(source: str) -> dict:
    with _index_lock:
        index = _load_index()
        kept = [c for c in index if c["source"] != source]
        _save_index(kept)
    return {"removed_chunks": len(index) - len(kept)}


TOOLS = {
    "extract_pdf": extract_pdf,
    "extract_pdf_bytes": extract_pdf_bytes,
    "add_chunks": add_chunks,
    "search": search,
    "stats": stats,
    "remove_source": remove_source,
}

# ─── JSON-RPC dispatch ───────────────────────────────────────────────


def _response(req_id, *, result=None, error=None) -> dict:
    out = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    return out


def _handle(msg: dict) -> dict | None:
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}
    if method == "initialize":
        proto = params.get("protocolVersion") or "1.1"
        return _response(req_id, result={
            "protocolVersion": proto if proto in ("1.1", "2.0") else "1.1",
            "serverInfo": {"name": MANIFEST["display_name"], "version": MANIFEST["version"]},
            "capabilities": {},
        })
    if method == "describe":
        return _response(req_id, result=MANIFEST)
    if method == "health":
        return _response(req_id, result={"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "version": MANIFEST["version"]})
    if method == "shutdown":
        return _response(req_id, result={"ok": True})
    if method == "invoke":
        tool = params.get("tool")
        fn = TOOLS.get(tool)
        if fn is None:
            return _response(req_id, error={"code": -32601, "message": f"Unknown tool: {tool}"})
        print(f"invoke: {tool}", file=sys.stderr, flush=True)
        try:
            data = fn(**(params.get("arguments") or {}))
        except Exception as e:  # noqa: BLE001
            print(f"{tool} failed: {e}", file=sys.stderr, flush=True)
            return _response(req_id, result={"success": False, "error": str(e)})
        return _response(req_id, result={"success": True, "tool": tool, "data": data})
    return _response(req_id, error={"code": -32601, "message": f"Method not found: {method}"})


def main() -> None:
    # On Windows the pipes default to the ANSI code page (cp1252). The host
    # reads frames as UTF-8 and silently drops any line that fails to
    # decode, so force UTF-8 both ways. Frames are also emitted as pure
    # ASCII (json.dumps default) as a second line of defence.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    print("cyberai-study-agent executa started", file=sys.stderr, flush=True)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps(_response(None, error={"code": -32700, "message": "Parse error"})), flush=True)
            continue
        if "method" not in msg:
            continue  # no reverse-RPCs are issued, so no responses to route
        resp = _handle(msg)
        if msg.get("id") is not None and resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
