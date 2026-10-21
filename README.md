Tessera
A local AI agent that runs on a laptop GPU. Gemma 4 E2B does the thinking. Python does the work. Memory lives in a small SQLite graph instead of a heavy RAG stack. No API keys, no cloud, no monthly bill.
Built and tested on a Windows laptop with an RTX 3050 (6 GB VRAM).

---
What it can do
Chat, in the browser (`python -m tessera.web`) or in a terminal (`python -m tessera`).
Build files you ask for. Web pages, Python scripts, JSON, Markdown. It writes them in parts, checks each part, fixes its own bugs and saves the result to `~/Documents/Tessera`.
Read your files. Name a file in your message and Tessera reads it before answering instead of guessing.
Remember things across sessions, and recall them by association, not just by similar wording.
Make new tools for itself. When it's missing a skill, it writes a small Python tool, tests it in a separate process and keeps it if it works.
Ask before risky actions, such as overwriting a file, installing a package or running generated code.
Things it has built end to end on the 3050: a live weather page using the Open-Meteo API, a peer-to-peer chat program in Python, and its own chat page.
---
How it works
1. A sequential agent pipeline
Every message goes through the same small set of steps, and only as many as it needs:
```
            ┌──────────┐
  message → │  Router  │  "Instant" / "Medium" / "Complex"
            └────┬─────┘
                 ▼
   interpret → develop → evaluate → respond
               (tools)   (checks the saved files)
```
Router. A grammar-constrained call decides how hard the request is. A greeting skips straight to `respond`; a "build me a website" request gets the full pass.
Develop. The model calls tools (`create_file`, file reading, skills) and sees what they return.
Evaluate. Judges the files that were actually saved, not what the model said it did, and can send the work back for revision.
One model, many roles. Every stage uses the same loaded Gemma with a different prompt, which is what makes this fit in 6 GB.
Pipelines are plain data in `tessera.toml`, so you can reorder or drop stages without touching code.
2. Memory in three layers
Layer	What it holds	Lifetime
Working memory	the current task: tool results, saved files, retry counters	one request
Conversation window	the last few messages	the session
Long-term memory	everything worth keeping, in SQLite, linked as an association graph	permanent
The long-term layer is inspired by how smell works. The smell of a biscuit can take you straight back to a childhood place, even though the smell itself has nothing to do with that place; the two are just connected.
So instead of only searching for text that looks like your question, Tessera:
finds a few starting points with small local embeddings (MiniLM, on the CPU),
spreads activation along the links in the graph (this → reminded me of that → which was connected to something else),
returns what lit up, including memories that share almost no words with the question.
One of the tests checks exactly this: cookie → bakery → childhood → grandmother is recalled even though "cookie" and "grandmother" have close to zero similarity.
It stays light: one SQLite file, a ~70 MB embedding model on the CPU, no vector database server.
3. Self-checking file builder
Small models make small mistakes, so Tessera checks everything before saving:
Big files are written section by section, each checked on its own, then the whole file is checked.
Syntax checks for HTML, CSS, JavaScript (`node --check`), JSON and Python.
Deterministic fixes, applied without asking the model again: duplicated functions are removed, the start-up call is moved below the code it calls, broken CSS is repaired, and hyphenated JavaScript names (`chat-history` → `chatHistory`) are renamed.
Pinpointed errors. When a check fails, the model sees the failing line with a pointer and a plain-English hint, not just "SyntaxError (line 1)".
Requirement checks. If you ask for `load_config` and `StaticApprover`, the file must actually use them. Code that calls itself "simulated" or "dummy" is rejected.
Context recovery. When a file outgrows the 4k context window, generation continues in a fresh context from a checkpoint.
---
Quick start (Windows)
You need: Python 3.11+, an NVIDIA GPU with 6 GB or more, the CUDA toolkit, and Node.js (optional, for JavaScript checks).
```powershell
git clone https://github.com/arnob58/main_tes.git tessera
cd tessera
python -m venv .venv
.venv\Scripts\activate

# llama-cpp-python with CUDA
$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --no-cache-dir "llama-cpp-python>=0.3.36"
pip install -e ".[memory,dev]"
```
Model: download a Gemma 4 E2B instruct GGUF (Q4_K_M fits comfortably in 6 GB) into `models/`.
Config:
```powershell
copy tessera.example.toml tessera.toml
```
Then set `model_path` in `tessera.toml` to your file.
Run:
```powershell
python -m tessera.web        # browser chat, opens http://localhost:8000
python -m tessera            # terminal chat
python -m tessera.web --scripted   # try the UI without loading a model
```
Run only one of them at a time, since there's room for one copy of the model on a 6 GB card.
On Linux/macOS it's the same steps, with `source .venv/bin/activate`, `CMAKE_ARGS="-DGGML_CUDA=on" pip install …` and `cp`.
Try these
```
Build a simple weather page for Toronto using the Open-Meteo API and save it as weather/weather.html
Read D:\tessera\README.md and summarise it in five bullet points
/remember I prefer dark-mode designs      (terminal only: commands like /remember, /recall, /skills)
```
---
Project layout
```
tessera/
  core/        orchestrator, task state, events, approvals, metrics
  agents/      the pipeline and how model output is parsed into tool calls
  llm/         llama.cpp engine, llama-server backend, router, prompts
  memory/      SQLite store, embeddings, association graph, semantic cache
  context/     token tracking, checkpoints, fresh-context continuation
  tools/       file builder, validators, requirement checks, code generation
  skills/      skill discovery, sandboxed runner, skill generation
  execution/   child-process harness for skills
  web/         browser UI (python -m tessera.web)
  ui/          terminal UI (python -m tessera)
tests/         219 tests; run without a GPU using a scripted model
```
`ARCHITECTURE.md` covers the design in depth. The short version: every part talks to the others through small interfaces, `app.py` is the only place they're wired together, and a test enforces which layers may import which.
Tests
```powershell
python -m pytest -q
```
No GPU needed. The tests use a scripted model plus real SQLite, real child processes and real local HTTP servers. Many of them replay actual mistakes Gemma made during development (a page split in the wrong places, a server that faked its replies, hyphenated JavaScript names), so each bug that showed up once is guarded against for good.
Honest limitations
It's a 2B model. It gets simple-to-medium tasks right and makes beginner mistakes on bigger ones. The checks catch many of them, not all. A larger multi-file program (for example, a web server wired into Tessera itself) still needed a human to finish.
It's slow. Expect a few tokens per second on a 3050, and a multi-step build can take several minutes.
Generated skills run as your user. There's a risk scan and an approval step, but it is not a security sandbox. Read what it asks to run.
Desktop control and vision (PyAutoGUI and Gemma vision) are implemented but early, and less tested than the rest.
License
MIT. See `LICENSE`.
