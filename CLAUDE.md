# CLAUDE.md — LorAI Master Build Plan

## What Is This?

This file is the complete execution plan for building LorAI — an AI-first operating environment. Feed this to Claude Code and it will create repos, write code, build Docker images, test everything, and push to GitHub and Docker Hub.

**Run this from an empty directory:**
```bash
mkdir ~/lorai-workspace && cd ~/lorai-workspace
claude
```
Then say: "Read CLAUDE.md and execute the full plan step by step."

---

## Project Overview

**LorAI** = "Lore" (knowledge) + "AI" + "LoRA" (Low-Rank Adaptation)
- Website: getlorai.com
- Tagline: "All of AI. One Command."
- Port: **1842** (the year Ada Lovelace wrote the first algorithm)
- License: MIT
- Docker image: `getlorai/desktop`
- PyPI package: `lorai`
- GitHub org: `getlorai`

**What LorAI does:** One `pip install lorai` gives developers a local, free, OpenAI-compatible AI platform with 50+ tools — LLMs, image gen, video, voice, code, agents, RAG, vision — all running in a Docker container on port 1842.

---

## Directory Structure to Create

```
~/lorai-workspace/
├── lorai/                    # The Docker OS project (→ github.com/getlorai/lorai)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── LICENSE
│   ├── README.md
│   ├── config/
│   │   ├── openbox-rc.xml
│   │   ├── openbox-menu.xml
│   │   └── supervisord.conf
│   ├── scripts/
│   │   └── entrypoint.sh
│   └── src/
│       ├── gateway/
│       │   └── gateway.py         # FastAPI API gateway on port 1842
│       ├── ai-shell/
│       │   ├── lorai_shell.py     # Core AI engine
│       │   └── lorai_terminal.py  # Interactive CLI
│       ├── file-manager/
│       │   └── lorai_fs.py        # Semantic file system
│       └── desktop/
│           └── lorai_panel.py     # Desktop overlay
│
├── lorai-sdk/                # The Python SDK (→ github.com/getlorai/lorai-sdk)
│   ├── pyproject.toml
│   ├── LICENSE
│   ├── README.md
│   ├── lorai/
│   │   ├── __init__.py
│   │   ├── client.py             # LorAI(OpenAI) — extends OpenAI SDK
│   │   ├── docker.py             # Auto-pull, start, stop Docker container
│   │   └── cli/
│   │       └── __init__.py       # CLI: lorai start/stop/chat/status/pull/bench
│   ├── tests/
│   │   └── test_client.py
│   └── examples/
│       └── quickstart.py
│
└── CLAUDE.md                 # This file
```

---

## Execution Plan (Do These In Order)

### PHASE 1: Create the SDK (lorai-sdk/)

The SDK is the user's entry point. `pip install lorai` should be the ONLY thing a developer needs to do.

#### 1.1 Create lorai-sdk/pyproject.toml
- Package name: `lorai`
- Version: `0.1.0`
- Dependencies: `openai>=1.30.0`, `httpx>=0.27.0`
- Entry point: `lorai = "lorai.cli:main"`
- Author: LorAI Team <hello@getlorai.com>
- Homepage: https://getlorai.com
- Repository: https://github.com/getlorai/lorai-sdk

#### 1.2 Create lorai-sdk/lorai/__init__.py
```python
__version__ = "0.1.0"
PORT = 1842
from lorai.client import LorAI
__all__ = ["LorAI", "PORT"]
```

#### 1.3 Create lorai-sdk/lorai/docker.py
This is the magic. It handles:
- `is_docker_installed()` — checks if docker CLI exists
- `is_image_pulled()` — checks if getlorai/desktop image exists locally
- `is_container_running()` — checks if container named "lorai" is running
- `is_lorai_healthy()` — GET http://localhost:1842/api/health
- `pull_image()` — runs `docker pull getlorai/desktop:latest`
- `start_container(port=1842, vnc_port=6080, gpu=False)` — runs docker run with proper flags
- `stop_container()` — docker stop lorai && docker rm lorai
- `ensure_running(port=1842, gpu=False)` — orchestrates everything: check → pull → start → health check
- `status()` — returns dict with all states

Container run command:
```
docker run -d --name lorai \
  -p {port}:1842 \
  -p {vnc_port}:6080 \
  -v ~/.lorai/data:/data \
  -e LORAI_MODE=hybrid \
  -e LORAI_MODEL=phi3:mini \
  getlorai/desktop:latest
```

#### 1.4 Create lorai-sdk/lorai/client.py
`LorAI` class that extends `OpenAI` from the openai package.

Constructor:
- Accepts: base_url, api_key, port (default 1842), auto_start (default True), gpu (default False), default_model (default "auto")
- If auto_start=True, calls `ensure_running(port, gpu)` before calling super().__init__
- base_url defaults to `http://localhost:{port}/v1`
- api_key defaults to "not-needed"
- After super().__init__, initializes these service objects:
  - `self.image` = _ImageService(self)
  - `self.video` = _VideoService(self)
  - `self.voice` = _VoiceService(self)
  - `self.knowledge` = _KnowledgeService(self)
  - `self.agents` = _AgentsService(self)
  - `self.code` = _CodeService(self)
  - `self.vision` = _VisionService(self)
  - `self.lora` = _LoRAService(self)
  - `self.hub` = _HubService(self)

Simple chat shortcut method:
```python
def chat(self, message, *, model=None, system=None, lora=None, temperature=0.7, max_tokens=None, stream=False, json_mode=False) -> str:
```

Service classes — each wraps either OpenAI endpoints or LorAI-native endpoints:

| Service | OpenAI or Native | Key Methods |
|---------|-----------------|-------------|
| _ImageService | OpenAI /v1/images | generate(prompt, model, size, lora, save_to), edit(image_path, prompt) |
| _VideoService | Native /lorai/video | generate(prompt, model, duration, fps, image_path, save_to) |
| _VoiceService | OpenAI /v1/audio | transcribe(audio_path, model), speak(text, voice, save_to) |
| _KnowledgeService | Native /lorai/knowledge | ingest(source, collection), search(query, top_k), ask(question) |
| _AgentsService | Native /lorai/agents | run(task, agents, tools, max_steps), list_agents(), list_tools() |
| _CodeService | OpenAI + Native | generate(prompt, language, execute), review(code) |
| _VisionService | OpenAI multimodal | analyze(image_path, prompt), ocr(image_path) |
| _LoRAService | Native /lorai/lora | list(), load(name, base_model), unload(name) |
| _HubService | Native /lorai/hub | models(), pull(model), remove(model), status(), bench() |

Native endpoints use httpx to POST to `http://localhost:{port}/lorai/...`

#### 1.5 Create lorai-sdk/lorai/cli/__init__.py
CLI commands via entry point `lorai`:
- `lorai start [--gpu] [--port PORT]` — ensure_running
- `lorai stop` — stop_container
- `lorai status` — print status dict nicely
- `lorai chat "message"` — POST to /v1/chat/completions, print result
- `lorai desktop` — open http://localhost:6080 in browser
- `lorai pull <model>` — POST to /lorai/hub/pull
- `lorai bench` — POST to /lorai/hub/bench
- `lorai logs` — docker logs --tail 50 -f lorai
- `lorai version` — print version, port, url
- `lorai help` — show ASCII banner + usage

Banner:
```
  ██╗      ██████╗  ██████╗  █████╗ ██╗
  ██║     ██╔═══██╗██╔══██╗██╔══██╗██║
  ██║     ██║   ██║██████╔╝███████║██║
  ██║     ██║   ██║██╔══██╗██╔══██║██║
  ███████╗╚██████╔╝██║  ██║██║  ██║██║
  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝

  All of AI. One Command. Port 1842.
```

#### 1.6 Create tests/test_client.py
Write tests using pytest + unittest.mock:
- Test PORT == 1842
- Test LorAI inherits from OpenAI
- Test auto_start calls ensure_running
- Test auto_start=False skips ensure_running
- Test custom port passes through
- Test gpu flag passes through
- Test all 9 services exist with correct methods
- Test CLI entry point is registered
- Test docker.py functions exist

#### 1.7 Verify SDK
```bash
cd lorai-sdk
pip install -e .
pytest tests/ -v  # all tests must pass
lorai version     # must print version and port 1842
```

---

### PHASE 2: Create the Docker OS (lorai/)

This is the container that runs on port 1842 and provides the AI platform.

#### 2.1 Create Dockerfile
Base: `alpine:3.19`
Install:
- System: xvfb, x11vnc, openbox, novnc, websockify, fonts, bash, curl, git, supervisor, pcmanfm
- Python: python3, py3-pip
- Python packages from requirements.txt (use --break-system-packages)
- **Ollama**: Download and install Ollama binary for Linux amd64

Create user `lorai` with sudo.
Create directories: ~/Desktop, ~/Documents, ~/Downloads, /data/vectors, /data/models, /data/config

Copy config, src, scripts into /opt/lorai/

Environment:
```
DISPLAY=:1
VNC_PORT=5900
NOVNC_PORT=6080
LORAI_PORT=1842
LORAI_MODE=hybrid
LORAI_MODEL=phi3:mini
HOME=/home/lorai
```

EXPOSE 1842 6080

Entrypoint: /opt/lorai/scripts/entrypoint.sh

#### 2.2 Create requirements.txt
```
fastapi>=0.110.0
uvicorn>=0.29.0
httpx>=0.27.0
websockets>=12.0
chromadb>=0.5.0
watchdog>=4.0.0
rich>=13.7.0
pydantic>=2.7.0
```
Note: Do NOT include ollama, langchain, whisper, piper etc in Phase 1. Keep it minimal. Add them in later phases.

#### 2.3 Create src/gateway/gateway.py — THE CRITICAL FILE
This is the API gateway that makes everything work. FastAPI app on port 1842.

Two route groups:
1. `/v1/*` — OpenAI-compatible proxy to Ollama (which runs on localhost:11434 inside the container)
2. `/lorai/*` — LorAI-native endpoints

**OpenAI-compatible routes (proxy to Ollama):**
- `POST /v1/chat/completions` → forward to Ollama's /v1/chat/completions (Ollama has native OpenAI compatibility)
- `POST /v1/completions` → forward to Ollama
- `POST /v1/embeddings` → forward to Ollama
- `GET /v1/models` → forward to Ollama
- `POST /v1/images/generations` → return 501 "Install ComfyUI: lorai install comfyui" (Phase 2)
- `POST /v1/audio/transcriptions` → return 501 "Install Whisper: lorai install whisper" (Phase 2)
- `POST /v1/audio/speech` → return 501 "Install Piper: lorai install piper" (Phase 2)

**LorAI-native routes (Phase 1 basics):**
- `GET /api/health` → `{"status": "ok", "version": "0.1.0", "port": 1842}`
- `GET /lorai/hub/status` → system info (cpu, ram, gpu, loaded models, uptime)
- `GET /lorai/hub/models` → list Ollama models (GET http://localhost:11434/api/tags)
- `POST /lorai/hub/pull` → pull Ollama model (POST http://localhost:11434/api/pull)
- `POST /lorai/hub/remove` → delete model (DELETE http://localhost:11434/api/delete)
- `POST /lorai/hub/bench` → basic benchmark (measure inference speed)

**LorAI-native routes (return 501 for Phase 2+):**
- `POST /lorai/video/generate` → 501
- `POST /lorai/knowledge/*` → 501
- `POST /lorai/agents/*` → 501
- `POST /lorai/lora/*` → 501
- `POST /lorai/code/execute` → 501

**Streaming support:**
The /v1/chat/completions endpoint MUST support streaming. When `stream: true` is in the request body, use `StreamingResponse` to proxy Ollama's SSE stream directly to the client.

**CORS:**
Enable CORS for all origins (local development tool).

#### 2.4 Create scripts/entrypoint.sh
Startup order:
1. Print LorAI ASCII banner with port info
2. Start Xvfb (virtual display)
3. Start Openbox (window manager)
4. Start x11vnc (VNC server)
5. Start websockify/noVNC (browser access on port 6080)
6. Start Ollama server (`ollama serve &`)
7. Wait for Ollama to be ready (poll localhost:11434 until responding)
8. Pull default model (`ollama pull phi3:mini`) — only if not already pulled
9. Start LorAI Gateway (`python3 /opt/lorai/src/gateway/gateway.py &`)
10. Start LorAI Shell
11. Start LorAI Desktop Panel
12. Launch LorAI Terminal on desktop
13. Print "Ready" message with both URLs
14. `wait` to keep container alive

#### 2.5 Create config/supervisord.conf (optional)
Alternative to shell-based process management. Can use this instead of manual & in entrypoint.

#### 2.6 Create remaining source files
- `src/ai-shell/lorai_shell.py` — FastAPI backend that takes natural language, queries Ollama, returns structured actions
- `src/ai-shell/lorai_terminal.py` — Rich-based interactive terminal, connects to shell backend via HTTP
- `src/file-manager/lorai_fs.py` — ChromaDB-backed semantic file indexer, watches ~/Documents, ~/Desktop, ~/Downloads
- `src/desktop/lorai_panel.py` — Sets wallpaper, keeps running
- `config/openbox-rc.xml` — Super key opens LorAI Terminal
- `config/openbox-menu.xml` — Right-click menu with LorAI Shell, File Manager, Terminal

#### 2.7 Build and Test Docker Image
```bash
cd lorai
docker build -t getlorai/desktop:latest .
```
Fix any build errors. Common issues:
- Alpine package names differ from Ubuntu (e.g., `py3-pip` not `python3-pip`)
- Ollama binary needs to be downloaded for the right architecture
- novnc path may be `/usr/share/novnc` or `/usr/share/webapps/novnc`

#### 2.8 Run and Verify
```bash
docker run -d --name lorai-test \
  -p 1842:1842 \
  -p 6080:6080 \
  getlorai/desktop:latest

# Wait 30 seconds for startup

# Test 1: Health check
curl http://localhost:1842/api/health
# Expected: {"status": "ok", "version": "0.1.0", "port": 1842}

# Test 2: List models
curl http://localhost:1842/v1/models
# Expected: JSON with phi3:mini listed

# Test 3: Chat
curl http://localhost:1842/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3:mini", "messages": [{"role": "user", "content": "Say hello in one sentence"}]}'
# Expected: JSON with assistant response

# Test 4: Streaming
curl http://localhost:1842/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "phi3:mini", "messages": [{"role": "user", "content": "Count to 5"}], "stream": true}'
# Expected: SSE stream of chunks

# Test 5: Desktop (open in browser)
# http://localhost:6080 should show Openbox desktop with LorAI terminal

# Clean up
docker stop lorai-test && docker rm lorai-test
```

---

### PHASE 3: Integration Test (SDK ↔ Docker)

#### 3.1 Start the container
```bash
docker run -d --name lorai -p 1842:1842 -p 6080:6080 getlorai/desktop:latest
# Wait for health check
```

#### 3.2 Test SDK against live container
```bash
cd lorai-sdk
pip install -e .
python -c "
from lorai import LorAI
ai = LorAI(auto_start=False)  # container already running
print(ai.chat('Say hello in one sentence'))
print('SUCCESS')
"
```
This must print a response from the LLM and "SUCCESS".

#### 3.3 Test CLI against live container
```bash
lorai status
# Should show: container running, API healthy

lorai chat "What is 2+2?"
# Should print: 4 (or similar)
```

#### 3.4 Test full auto-start flow
```bash
docker stop lorai && docker rm lorai
python -c "
from lorai import LorAI
ai = LorAI()  # should auto-start the container
print(ai.chat('Hello!'))
"
```

---

### PHASE 4: Push Everything

#### 4.1 Push SDK to GitHub
```bash
cd lorai-sdk
git init
git add .
git commit -m "feat: LorAI SDK v0.1.0 — All of AI. One Command. Port 1842.

- LorAI(OpenAI) client with 10 AI services
- Auto Docker pull/start on first use
- CLI: lorai start/stop/chat/status/pull/bench
- OpenAI-compatible + LorAI-native endpoints
- 21 tests passing"

git remote add origin https://github.com/getlorai/lorai-sdk.git
git branch -M main
git push -u origin main
```

#### 4.2 Push OS to GitHub
```bash
cd lorai
git init
git add .
git commit -m "feat: LorAI OS v0.1.0 — AI-first operating environment

- Docker-native desktop accessible via browser
- Ollama + OpenAI-compatible API gateway on port 1842
- LorAI Shell (natural language to system commands)
- Semantic file manager with ChromaDB
- noVNC browser desktop on port 6080"

git remote add origin https://github.com/getlorai/lorai.git
git branch -M main
git push -u origin main
```

#### 4.3 Push Docker Image
```bash
docker login
docker push getlorai/desktop:latest
```

#### 4.4 Publish SDK to PyPI
```bash
cd lorai-sdk
pip install build twine
python -m build
twine upload dist/*
```

---

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Port | 1842 | Ada Lovelace's year. Unique, memorable, tells a story |
| Base image | Alpine 3.19 | ~5MB base, minimal attack surface |
| LLM server | Ollama | Best DX, OpenAI-compatible out of the box |
| Default model | phi3:mini | Runs on CPU, small, fast, good quality |
| API gateway | FastAPI | Async, fast, auto-docs, Python ecosystem |
| SDK base | OpenAI Python SDK | Zero learning curve for developers |
| Vector DB | ChromaDB | Embedded, zero config, persistent |
| Desktop | Openbox + noVNC | Lightweight, browser-accessible |
| Process mgmt | Shell script (entrypoint.sh) | Simple, debuggable, Docker-native |

## API Contract

```
http://localhost:1842/
├── v1/                          # OpenAI-compatible (proxied to Ollama)
│   ├── chat/completions         # POST - Chat (streaming supported)
│   ├── completions              # POST - Text completion
│   ├── embeddings               # POST - Embeddings
│   └── models                   # GET  - List models
│
├── lorai/                       # LorAI-native
│   ├── hub/
│   │   ├── status               # GET  - System status
│   │   ├── models               # GET  - List models (detailed)
│   │   ├── pull                 # POST - Download model
│   │   ├── remove               # POST - Delete model
│   │   └── bench                # POST - Benchmark hardware
│   ├── video/generate           # POST - Text/image to video (Phase 2)
│   ├── knowledge/
│   │   ├── ingest               # POST - Ingest documents (Phase 2)
│   │   ├── search               # POST - Semantic search (Phase 2)
│   │   └── ask                  # POST - RAG Q&A (Phase 2)
│   ├── agents/
│   │   ├── run                  # POST - Run agent workflow (Phase 2)
│   │   ├── list                 # GET  - List agents (Phase 2)
│   │   └── tools                # GET  - List MCP tools (Phase 2)
│   ├── lora/
│   │   ├── list                 # GET  - List LoRA adapters (Phase 3)
│   │   ├── load                 # POST - Load LoRA (Phase 3)
│   │   └── unload               # POST - Unload LoRA (Phase 3)
│   └── code/execute             # POST - Execute code (Phase 2)
│
└── api/health                   # GET  - Health check
```

## Success Criteria

After completing all phases, this must work:

```bash
# From a fresh machine with only Docker and Python installed:
pip install lorai

python -c "
from lorai import LorAI
ai = LorAI()           # auto-pulls Docker image, starts container
print(ai.chat('Hello'))  # gets response from local LLM
ai.stop()              # stops container
"
```

And this must work:
```bash
lorai start
lorai chat "What is LorAI?"
lorai status
lorai desktop    # opens browser to noVNC desktop
lorai stop
```

---

## Notes for Claude Code

- Focus on getting Phase 1 and 2 working end-to-end BEFORE moving to advanced features
- The API gateway (gateway.py) is the most critical file — it's the bridge between SDK and Ollama
- Test each step before moving to the next
- If Alpine packages fail, try alternative names or build from source
- Ollama binary for Linux: `curl -fsSL https://ollama.com/install.sh | sh` or download directly
- If novnc path is wrong, find it with `find / -name "vnc.html" 2>/dev/null`
- The first `ollama pull phi3:mini` inside the container will take time — that's expected
- Stream proxying is tricky — make sure to use `StreamingResponse` with async generators
- CORS must be enabled or browser-based clients won't work
