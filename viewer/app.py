"""Mobile-friendly lesson viewer: read a lesson (top pane), run its code and
watch live output (bottom pane, independently scrollable), and ask questions
via a floating modal — answered in the background by the Claude API and
shown after a refresh. Not part of the curriculum itself, just infra for
reading this repo comfortably from a phone over Tailscale."""
import asyncio
import json
import re
import time
import uuid
from pathlib import Path

import anthropic
import markdown as md
from fastapi import BackgroundTasks, FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

_PYGMENTS_LIGHT = HtmlFormatter(style="default").get_style_defs(".highlight")
_PYGMENTS_DARK = HtmlFormatter(style="monokai").get_style_defs(".highlight")
PYGMENTS_CSS = (
    f"<style>{_PYGMENTS_LIGHT}\n"
    f"@media (prefers-color-scheme: dark) {{ {_PYGMENTS_DARK} }}</style>"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LESSONS_DIR = REPO_ROOT / "lessons"
CODE_DIR = LESSONS_DIR / "code"
QNA_DIR = Path(__file__).resolve().parent / "qna_data"
QNA_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"

SECRETS_FILE = Path.home() / ".config" / "secrets" / "secrets.env"


def _load_api_key() -> str | None:
    if not SECRETS_FILE.exists():
        return None
    for line in SECRETS_FILE.read_text().splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


_ANTHROPIC_KEY = _load_api_key()
_client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY) if _ANTHROPIC_KEY else None

app = FastAPI()

BASE_CSS = """
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font: 17px/1.55 -apple-system, system-ui, Roboto, sans-serif;
    background: Canvas; color: CanvasText;
  }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.15rem; margin-top: 1.6rem; }
  a { color: #4098ff; }
  code { background: rgba(128,128,128,0.18); padding: 0.1em 0.3em; border-radius: 4px; font-size: 0.9em; }
  pre.codeblock { background: rgba(128,128,128,0.12); padding: 0.8rem; border-radius: 8px;
        overflow-x: auto; font-size: 0.85em; }
  pre.codeblock code { background: none; padding: 0; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
  td, th { border: 1px solid rgba(128,128,128,0.35); padding: 0.4em 0.6em; }
  .nav { display: flex; justify-content: space-between; margin-bottom: 1rem; font-size: 0.95rem; }
  .content-wrap { padding: 1rem 1.1rem; }
  .runbtn, .askbtn {
    display: block; width: 100%; padding: 0.8rem; margin: 0.6rem 0 0;
    font-size: 1rem; font-weight: 600; text-align: center;
    background: #2f7de1; color: white; border: none; border-radius: 10px;
    text-decoration: none; cursor: pointer;
  }
  .runbtn:active, .askbtn:active { background: #2564bd; }
  ul.lessons { list-style: none; padding: 0; }
  ul.lessons li a {
    display: block; padding: 0.9rem 1rem; margin-bottom: 0.6rem;
    border-radius: 10px; background: rgba(128,128,128,0.12);
    text-decoration: none; color: inherit; font-size: 1.05rem;
  }

  /* split screen, draggable */
  .split { display: flex; flex-direction: column; height: 100dvh; }
  .pane-top { flex: none; height: 56vh; overflow-y: auto; min-height: 60px; }
  .resizer {
    flex: none; height: 22px; cursor: row-resize; touch-action: none;
    background: rgba(128,128,128,0.25);
    border-top: 1px solid rgba(128,128,128,0.4);
    border-bottom: 1px solid rgba(128,128,128,0.4);
    display: flex; align-items: center; justify-content: center;
  }
  .resizer::before {
    content: ""; width: 40px; height: 4px; border-radius: 2px;
    background: rgba(128,128,128,0.7);
  }
  .pane-bottom {
    flex: 1 1 auto; overflow-y: auto; min-height: 60px;
    background: rgba(128,128,128,0.04);
    padding: 0.8rem 1rem 1rem;
  }

  /* code viewer */
  details.codefile { margin: 0.8rem 0; }
  details.codefile summary {
    cursor: pointer; font-weight: 600; padding: 0.6rem 0.8rem;
    border-radius: 8px; background: rgba(128,128,128,0.12); list-style: none;
  }
  details.codefile summary::-webkit-details-marker { display: none; }
  details.codefile summary::before { content: "▸ "; }
  details.codefile[open] summary::before { content: "▾ "; }
  .srcfile { margin-top: 0.5rem; max-height: 60vh; overflow: auto; border-radius: 8px; }
  .srcfile .highlight { margin: 0; }
  .srcfile pre { margin: 0; padding: 0.8rem; font-size: 0.82em; line-height: 1.45; }
  #log {
    background: #0d1117; color: #d6e2ea; padding: 0.8rem; border-radius: 8px;
    white-space: pre-wrap; word-break: break-word; font-size: 0.8em;
    min-height: 2rem; margin-top: 0.6rem;
  }
  .status { font-size: 0.85rem; opacity: 0.75; margin-top: 0.4rem; }

  /* floating ask button + modal */
  .fab {
    position: fixed; right: 1rem; bottom: 1rem; z-index: 50;
    width: 3.2rem; height: 3.2rem; border-radius: 50%;
    background: #e14e9c; color: white; border: none; font-size: 1.4rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  }
  .modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
    z-index: 100; align-items: flex-end;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: Canvas; color: CanvasText; width: 100%; border-radius: 16px 16px 0 0;
    padding: 1.2rem; max-height: 80vh; overflow-y: auto;
  }
  .modal textarea {
    width: 100%; min-height: 6rem; font-size: 1rem; padding: 0.6rem;
    border-radius: 8px; border: 1px solid rgba(128,128,128,0.4);
    background: Field; color: FieldText; font-family: inherit;
  }
  .qna-item { margin: 0.8rem 0; padding: 0.7rem 0.9rem; border-radius: 8px;
              background: rgba(128,128,128,0.08); font-size: 0.92rem; }
  .qna-q { font-weight: 600; }
  .qna-a { margin-top: 0.4rem; white-space: pre-wrap; }
  .qna-pending { opacity: 0.7; font-style: italic; }
</style>
"""

MODAL_HTML = """
<button class="fab" onclick="document.getElementById('askModal').classList.add('open')">?</button>
<div class="modal-overlay" id="askModal">
  <div class="modal">
    <h2 style="margin-top:0">Ask a question</h2>
    <p style="font-size:0.85rem; opacity:0.75">
      Answered in the background by Claude, using this lesson's content as
      context. Close this and refresh the page in a bit to see the answer
      below the lesson text.
    </p>
    <form method="post" action="/ask/{n}" onsubmit="setTimeout(() => document.getElementById('askModal').classList.remove('open'), 50)">
      <textarea name="question" placeholder="What's confusing?" required></textarea>
      <button class="askbtn" type="submit">Submit</button>
    </form>
    <button class="askbtn" style="background:#666" onclick="document.getElementById('askModal').classList.remove('open')">Cancel</button>
  </div>
</div>
"""


def page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'><title>{title}</title>{BASE_CSS}{PYGMENTS_CSS}</head><body>{body}</body></html>"


def lesson_files():
    files = sorted(LESSONS_DIR.glob("[0-9][0-9]-*.md"))
    return {int(f.name[:2]): f for f in files}


def code_file_for(n: int):
    matches = list(CODE_DIR.glob(f"{n:02d}_*.py"))
    return matches[0] if matches else None


def code_block(title: str, path: Path) -> str:
    highlighted = highlight(path.read_text(), PythonLexer(), HtmlFormatter(cssclass="highlight"))
    return (
        f"<details class='codefile'><summary>{title}</summary>"
        f"<div class='srcfile'>{highlighted}</div></details>"
    )


def code_files_html(n: int) -> str:
    blocks = []
    main = code_file_for(n)
    if main:
        blocks.append(code_block(main.name, main))
    common = CODE_DIR / "common.py"
    if common.exists() and (main is None or common != main):
        blocks.append(code_block("common.py (shared, imported by every lesson)", common))
    if not blocks:
        return ""
    return f"<h2>Code</h2>{''.join(blocks)}"


def qna_path(n: int) -> Path:
    return QNA_DIR / f"lesson_{n}.json"


def load_qna(n: int):
    p = qna_path(n)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def save_qna(n: int, items):
    qna_path(n).write_text(json.dumps(items, indent=2))


def qna_html(n: int) -> str:
    items = load_qna(n)
    if not items:
        return ""
    rows = []
    for it in reversed(items):
        if it["status"] == "pending":
            rows.append(
                f"<div class='qna-item'><div class='qna-q'>Q: {it['question']}</div>"
                f"<div class='qna-a qna-pending'>&#8987; thinking&hellip; refresh to check</div></div>"
            )
        else:
            answer_html = md.markdown(it.get("answer") or "")
            rows.append(
                f"<div class='qna-item'><div class='qna-q'>Q: {it['question']}</div>"
                f"<div class='qna-a'>{answer_html}</div></div>"
            )
    return f"<h2>Questions & answers</h2>{''.join(rows)}"


async def _answer_question(n: int, entry_id: str):
    items = load_qna(n)
    entry = next((i for i in items if i["id"] == entry_id), None)
    if entry is None:
        return
    try:
        md_file = LESSONS_DIR / "00-roadmap.md" if n == 0 else lesson_files().get(n)
        code_file = code_file_for(n)
        context = md_file.read_text() if md_file else ""
        if code_file:
            context += "\n\n---CODE---\n" + code_file.read_text()

        if _client is None:
            raise RuntimeError("no ANTHROPIC_API_KEY found in secrets store")

        resp = _client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            system=(
                "You are a teaching assistant for a from-scratch nanoGPT/LLM-pretraining "
                "lesson repo. The reader has near-zero PyTorch background. House style, "
                "follow strictly: lead every explanation with a small CONCRETE example "
                "using real values, lined up/aligned so the relationship is visible "
                "without inference — analogy comes only after, and only if it adds "
                "something. Keep answers short and focused on exactly what was asked; "
                "do not dump unrelated background. Here is the lesson's markdown and code "
                "for context:\n\n" + context
            ),
            messages=[{"role": "user", "content": entry["question"]}],
        )
        answer = "".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:
        answer = f"(couldn't get an answer: {e})"

    items = load_qna(n)  # reload in case of concurrent writes
    for it in items:
        if it["id"] == entry_id:
            it["answer"] = answer
            it["status"] = "answered"
    save_qna(n, items)


@app.get("/", response_class=HTMLResponse)
def index():
    items = "".join(
        f"<li><a href='/lesson/{n}'>{f.stem}</a></li>"
        for n, f in sorted(lesson_files().items())
    )
    roadmap = LESSONS_DIR / "00-roadmap.md"
    roadmap_item = f"<li><a href='/lesson/0'>{roadmap.stem}</a></li>" if roadmap.exists() else ""
    visual_item = "<li><a href='/visual/params-kv'>&#129518; visual: counting params &amp; KV cache</a></li>"
    body = f"<div class='content-wrap'><h1>nanogpt-from-scratch</h1><ul class='lessons'>{roadmap_item}{items}{visual_item}</ul></div>"
    return page("Lessons", body)


@app.get("/visual/params-kv", response_class=HTMLResponse)
def visual_params_kv():
    return (STATIC_DIR / "visual-params-kv.html").read_text()


@app.get("/lesson/{n}", response_class=HTMLResponse)
def lesson(n: int):
    f = (LESSONS_DIR / "00-roadmap.md") if n == 0 else lesson_files().get(n)
    if f is None or not f.exists():
        return page("Not found", "<p>No such lesson.</p><a href='/'>&larr; back</a>")

    text = f.read_text()

    def fix_link(m):
        target = m.group(1)
        num = re.match(r"(\d\d)-([\w-]+)\.md(#[\w-]*)?", target)
        if not num:
            return m.group(0)
        fragment = num.group(3) or ""
        return f"(/lesson/{int(num.group(1))}{fragment})"

    text = re.sub(r"\((\d\d-[\w-]+\.md(?:#[\w-]*)?)\)", fix_link, text)
    html = md.markdown(text, extensions=["fenced_code", "tables", "toc"])
    html = html.replace("<pre>", "<pre class='codeblock'>")

    code = code_file_for(n)
    nav = "<div class='nav'><a href='/'>&larr; all lessons</a></div>"
    visual_banner = (
        "<p><a href='/visual/params-kv' class='runbtn' style='display:inline-block;width:auto;padding:0.6rem 1rem;'>"
        "&#129518; Visual: counting these params &amp; KV cache</a></p>"
        if n == 6 else ""
    )
    top_pane = f"<div class='content-wrap'>{nav}{html}{visual_banner}{code_files_html(n)}{qna_html(n)}</div>"

    if code:
        bottom_pane = f"""
        <button class="runbtn" onclick="startRun()">&#9654; Run {code.name}</button>
        <div class="status" id="status"></div>
        <pre id="log"></pre>
        <script>
          function startRun() {{
            const log = document.getElementById('log');
            const status = document.getElementById('status');
            log.textContent = '';
            status.textContent = 'starting…';
            const es = new EventSource('/stream/{n}');
            es.onopen = () => status.textContent = 'running…';
            es.onmessage = (e) => {{
              log.textContent += e.data + '\\n';
              log.scrollTop = log.scrollHeight;
            }};
            es.addEventListener('done', () => {{ status.textContent = 'finished'; es.close(); }});
            es.onerror = () => {{ status.textContent = 'connection closed'; es.close(); }};
          }}
        </script>
        """
    else:
        bottom_pane = "<div class='status'>No runnable code for this lesson.</div>"

    body = f"""
    <div class="split" id="split">
      <div class="pane-top" id="paneTop">{top_pane}</div>
      <div class="resizer" id="resizer"></div>
      <div class="pane-bottom" id="paneBottom">{bottom_pane}</div>
    </div>
    {MODAL_HTML.format(n=n)}
    <script>
      (function() {{
        const resizer = document.getElementById('resizer');
        const topPane = document.getElementById('paneTop');
        let dragging = false, startY = 0, startHeight = 0;
        function begin(y) {{
          dragging = true; startY = y;
          startHeight = topPane.getBoundingClientRect().height;
        }}
        function move(y) {{
          if (!dragging) return;
          const maxH = window.innerHeight - 60;
          let newH = startHeight + (y - startY);
          newH = Math.max(60, Math.min(maxH, newH));
          topPane.style.height = newH + 'px';
        }}
        function end() {{ dragging = false; }}
        resizer.addEventListener('mousedown', e => begin(e.clientY));
        window.addEventListener('mousemove', e => move(e.clientY));
        window.addEventListener('mouseup', end);
        resizer.addEventListener('touchstart', e => {{ begin(e.touches[0].clientY); e.preventDefault(); }}, {{passive: false}});
        window.addEventListener('touchmove', e => {{ if (dragging) {{ move(e.touches[0].clientY); e.preventDefault(); }} }}, {{passive: false}});
        window.addEventListener('touchend', end);
      }})();
    </script>
    """
    return page(f.stem, body)


@app.post("/ask/{n}")
async def ask(n: int, background_tasks: BackgroundTasks, question: str = Form(...)):
    items = load_qna(n)
    entry_id = uuid.uuid4().hex[:8]
    items.append({"id": entry_id, "question": question, "answer": None, "status": "pending", "ts": time.time()})
    save_qna(n, items)
    background_tasks.add_task(_answer_question, n, entry_id)
    return RedirectResponse(url=f"/lesson/{n}", status_code=303)


@app.get("/stream/{n}")
async def stream(n: int):
    code = code_file_for(n)

    async def gen():
        if code is None:
            yield "data: no code file for this lesson\n\n"
            return
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "python", "-u", str(code.relative_to(REPO_ROOT)),
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            text = line.decode(errors="replace").rstrip("\n")
            for part in (text.split("\n") if text else [""]):
                yield f"data: {part}\n\n"
        await proc.wait()
        yield "event: done\ndata: exit\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
