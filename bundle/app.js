// CyberAI Study Agent — app UI.
//
// Model calls (embeddings and answers) go from here straight to the Anna
// host via anna.llm.*; the bundled Executa handles everything local: PDF
// text extraction, the vector index on disk, similarity search and prompt
// assembly.
import { AnnaAppRuntime } from "/static/anna-apps/_sdk/latest/index.js";

const TOOL_ID = "tool-dev-cyberai-study-agent";
const EMBED_BATCH = 16;
const MAX_PDF_BYTES = 10 * 1024 * 1024; // base64 must fit the host frame limit
const HISTORY_TURNS = 6;

const $ = (id) => document.getElementById(id);
const state = { anna: null, history: [], docs: [], busy: false };

// ─── Helpers ──────────────────────────────────────────────────────────

const tool = (method, args) => state.anna.tools.invoke({ tool_id: TOOL_ID, method, args });

async function embed(texts) {
  const out = await state.anna.llm.embed({ input: texts });
  return [...(out?.data || [])]
    .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
    .map((d) => d.embedding);
}

function textOf(result) {
  const c = result?.content;
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map((b) => b?.text || "").join("");
  if (c && typeof c === "object") return c.text || "";
  return result?.text || "";
}

function errorText(e) {
  const msg = e?.message || String(e);
  if (/timed out/i.test(msg)) return "The request took too long. Try again, or split very large PDFs.";
  return msg;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",", 2)[1] || "");
    r.onerror = () => reject(r.error || new Error("Could not read the file."));
    r.readAsDataURL(file);
  });
}

// ─── Library ──────────────────────────────────────────────────────────

async function refreshLibrary() {
  const data = await tool("stats", {});
  state.docs = data.sources || [];
  const list = $("doc-list");
  list.replaceChildren();
  for (const d of state.docs) {
    const li = document.createElement("li");
    li.className = "doc";
    li.innerHTML = `
      <span class="doc-name">${escapeHtml(d.source)}</span>
      <span class="doc-meta">${d.course ? escapeHtml(d.course) + ", " : ""}${d.pages} pages</span>
      <button class="btn btn-quiet doc-remove" type="button">Remove</button>`;
    const rm = li.querySelector("button");
    rm.addEventListener("click", () => removeDoc(d.source, rm));
    list.append(li);
  }
  $("doc-empty").hidden = state.docs.length > 0;

  const courses = [...new Set(state.docs.map((d) => d.course).filter(Boolean))].sort();
  const filter = $("course-filter");
  const current = filter.value;
  filter.replaceChildren(new Option("All courses", ""), ...courses.map((c) => new Option(c, c)));
  filter.value = courses.includes(current) ? current : "";
  $("course-list").replaceChildren(...courses.map((c) => new Option(c)));
}

async function removeDoc(source, button) {
  // Two-step confirm in place: dialogs may be blocked inside the app frame.
  if (button.dataset.armed !== "1") {
    button.dataset.armed = "1";
    button.textContent = "Confirm";
    setTimeout(() => { button.dataset.armed = ""; button.textContent = "Remove"; }, 4000);
    return;
  }
  try {
    await tool("remove_source", { source });
    await refreshLibrary();
  } catch (e) {
    setAddStatus(errorText(e), true);
  }
}

function setAddStatus(text, isError = false) {
  const el = $("add-status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

async function addFiles(files) {
  const course = $("course-input").value.trim();
  const btn = $("add-btn");
  btn.disabled = true;
  try {
    for (const file of files) {
      if (file.size > MAX_PDF_BYTES) {
        setAddStatus(`${file.name} is larger than 10 MB. Split it into smaller PDFs and add them one by one.`, true);
        continue;
      }
      setAddStatus(`Reading ${file.name}…`);
      const doc = await tool("extract_pdf_bytes", { name: file.name, data: await fileToBase64(file), course });
      const chunks = doc.chunks;
      for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
        setAddStatus(`Indexing ${file.name}: ${Math.min(i + EMBED_BATCH, chunks.length)} of ${chunks.length} passages`);
        const batch = chunks.slice(i, i + EMBED_BATCH);
        const vectors = await embed(batch.map((c) => c.text));
        await tool("add_chunks", {
          source: doc.source,
          course: doc.course,
          chunks: batch.map((c, j) => ({ ...c, vector: vectors[j] })),
          replace: i === 0,
        });
      }
      setAddStatus(`Added ${doc.source} (${doc.pages} pages).`);
      await refreshLibrary();
    }
  } catch (e) {
    setAddStatus(errorText(e), true);
  } finally {
    btn.disabled = false;
    $("file-input").value = "";
  }
}

// ─── Answers and citations ────────────────────────────────────────────

// Matches [file.pdf, p.10] and [file.pdf, pp. 10, 12-14].
const CITE_RE = /\[([^\[\]]+?\.pdf),\s*pp?\.\s*([\d\s,\-–]+)\]/gi;

function pagesOf(spec) {
  const pages = [];
  for (const part of spec.split(",")) {
    const [a, b] = part.split(/[-–]/).map((n) => parseInt(n, 10));
    if (Number.isNaN(a)) continue;
    if (!Number.isNaN(b) && b >= a && b - a < 50) for (let p = a; p <= b; p++) pages.push(p);
    else pages.push(a);
  }
  return pages;
}

function inline(text) {
  // text is already HTML-escaped
  return text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, "$1<em>$2</em>");
}

// Fallback for answers that still contain LaTeX: drop the $ delimiters
// and turn the most common commands into Unicode.
const TEX = {
  "\\delta": "δ", "\\Delta": "Δ", "\\sigma": "σ", "\\Sigma": "Σ", "\\alpha": "α", "\\beta": "β",
  "\\gamma": "γ", "\\lambda": "λ", "\\varepsilon": "ε", "\\epsilon": "ε", "\\in": "∈", "\\notin": "∉",
  "\\subseteq": "⊆", "\\subset": "⊂", "\\cup": "∪", "\\cap": "∩", "\\times": "×", "\\to": "→",
  "\\rightarrow": "→", "\\leftarrow": "←", "\\leq": "≤", "\\geq": "≥", "\\neq": "≠", "\\forall": "∀",
  "\\exists": "∃", "\\emptyset": "∅", "\\cdot": "·", "\\ldots": "…", "\\dots": "…",
};
const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };

function untex(text) {
  return text.replace(/\$\$?([^$]+?)\$\$?/g, (_, m) => {
    let t = m;
    for (const [k, v] of Object.entries(TEX)) t = t.replace(new RegExp(k.replace(/\\/g, "\\\\") + "(?![a-zA-Z])", "g"), v);
    t = t
      .replace(/\\(?:bar|overline|hat|tilde)\{([^}]*)\}/g, "$1\u0304")
      .replace(/\\(?:mathcal|mathbf|mathrm|text)\{([^}]*)\}/g, "$1")
      .replace(/_\{(\d+)\}|_(\d+)/g, (_, a, b) => [...(a || b)].map((c) => SUB[c]).join(""))
      .replace(/\\([{}])/g, "$1")
      .replace(/_\{([^}]*)\}/g, "_$1")
      .replace(/\\,|\\;|\\ /g, " ");
    return t;
  });
}

function renderAnswer(answer, sources) {
  const lines = escapeHtml(untex(answer)).split("\n");
  let html = "";
  let para = [];
  let depth = 0; // open <ul> levels (0, 1 or 2)
  let baseIndent = null;
  const flush = () => {
    if (para.length) html += `<p>${inline(para.join(" "))}</p>`;
    para = [];
  };
  const closeTo = (d) => {
    while (depth > d) {
      html += depth === 2 ? "</li></ul>" : "</li></ul>";
      depth--;
    }
  };
  for (const raw of lines) {
    const item = raw.match(/^(\s*)(?:[*\-•]|\d+\.)\s+(.*)$/);
    if (item) {
      flush();
      const indent = item[1].replace(/\t/g, "    ").length;
      if (depth === 0) baseIndent = indent;
      const want = indent > baseIndent + 1 ? 2 : 1;
      if (depth === 0) { html += "<ul><li>"; depth = 1; if (want === 2) { html += "<ul><li>"; depth = 2; } }
      else if (want > depth) { html += "<ul><li>"; depth = 2; }
      else if (want < depth) { closeTo(want); html += "</li><li>"; }
      else html += "</li><li>";
      html += inline(item[2]);
      continue;
    }
    const line = raw.trim();
    if (!line) { flush(); continue; }
    if (depth) { closeTo(0); }
    para.push(line);
  }
  flush();
  closeTo(0);

  // Turn citations into highlighter buttons, one per page.
  return html.replace(CITE_RE, (_, file, spec) => {
    const name = file.trim();
    return pagesOf(spec)
      .map((page) => {
        const hit = sources.some((s) => s.source === name && s.page === page);
        return `<button type="button" class="cite${hit ? "" : " unmatched"}" data-source="${name}" data-page="${page}" aria-pressed="false"${hit ? "" : ' title="This page was not among the retrieved passages"'}>${name.replace(/\.pdf$/i, "")}, p.${page}</button>`;
      })
      .join("");
  });
}

function showExcerpt(source, page, sources, button) {
  const passages = sources.filter((s) => s.source === source && s.page === page);
  if (!passages.length) return;
  document.querySelectorAll(".cite[aria-pressed='true']").forEach((b) => b.setAttribute("aria-pressed", "false"));
  button.setAttribute("aria-pressed", "true");
  $("excerpt-source").textContent = source;
  $("excerpt-page").textContent = `Page ${page}`;
  const body = $("excerpt-body");
  const extra = body.parentElement.querySelectorAll(".excerpt-body:not(#excerpt-body)");
  extra.forEach((n) => n.remove());
  body.textContent = passages[0].text;
  for (const p of passages.slice(1)) {
    const more = document.createElement("div");
    more.className = "excerpt-body";
    more.textContent = p.text;
    body.after(more);
  }
  $("excerpt").hidden = false;
  document.querySelector(".app").classList.add("with-excerpt");
}

function closeExcerpt() {
  $("excerpt").hidden = true;
  document.querySelector(".app").classList.remove("with-excerpt");
  document.querySelectorAll(".cite[aria-pressed='true']").forEach((b) => b.setAttribute("aria-pressed", "false"));
}

async function ask(question) {
  $("welcome").hidden = true;
  const turn = document.createElement("section");
  turn.className = "turn";
  turn.innerHTML = `<p class="turn-q"></p><div class="turn-a pending">Searching your PDFs…</div>`;
  turn.querySelector(".turn-q").textContent = question;
  $("thread").append(turn);
  turn.scrollIntoView({ block: "end" });
  const out = turn.querySelector(".turn-a");

  try {
    const [qvec] = await embed([question]);
    const found = await tool("search", {
      vector: qvec,
      question,
      course: $("course-filter").value,
      history: state.history.slice(-HISTORY_TURNS * 2),
    });
    if (!found.prompt) {
      out.className = "turn-a";
      out.textContent = "There is nothing to search yet. Add a PDF from the panel on the left first.";
      return;
    }
    out.textContent = "Writing the answer…";
    const res = await state.anna.llm.complete({
      messages: [{ role: "user", content: { type: "text", text: found.prompt } }],
      systemPrompt: found.system_prompt,
      maxTokens: 1500,
      temperature: 0.2,
    });
    const answer = textOf(res).trim() || "The model returned an empty answer. Try rephrasing the question.";
    out.className = "turn-a";
    out.innerHTML = renderAnswer(answer, found.sources);
    out.querySelectorAll(".cite:not(.unmatched)").forEach((b) =>
      b.addEventListener("click", () => showExcerpt(b.dataset.source, Number(b.dataset.page), found.sources, b))
    );
    state.history.push({ role: "user", text: question }, { role: "assistant", text: answer });
  } catch (e) {
    out.className = "turn-a error";
    out.textContent = errorText(e);
  }
}

// ─── Wiring ───────────────────────────────────────────────────────────

async function main() {
  try {
    state.anna = await AnnaAppRuntime.connect();
  } catch {
    $("doc-empty").textContent = "Open this app inside Anna to use it.";
    return;
  }
  await state.anna.window.set_title({ title: "CyberAI Study Agent" });

  $("add-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => {
    const files = [...e.target.files];
    if (files.length) addFiles(files);
  });

  const q = $("question");
  const grow = () => { q.style.height = "auto"; q.style.height = q.scrollHeight + "px"; };
  q.addEventListener("input", grow);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask-form").requestSubmit(); }
  });

  $("ask-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = q.value.trim();
    if (!question || state.busy) return;
    state.busy = true;
    $("ask-btn").disabled = true;
    q.value = "";
    grow();
    try { await ask(question); } finally { state.busy = false; $("ask-btn").disabled = false; q.focus(); }
  });

  $("new-chat").addEventListener("click", () => {
    state.history = [];
    $("thread").querySelectorAll(".turn").forEach((t) => t.remove());
    $("welcome").hidden = false;
    closeExcerpt();
  });
  $("excerpt-close").addEventListener("click", closeExcerpt);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeExcerpt(); });

  try {
    await refreshLibrary();
  } catch (e) {
    setAddStatus(errorText(e), true);
  }
}

main();
