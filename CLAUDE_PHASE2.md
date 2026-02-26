# CLAUDE_PHASE2.md — LorAI Full Services Implementation

## Prerequisites

Phase 1 (CLAUDE.md) must be complete:
- Docker container running on port 1842
- Ollama serving LLMs, gateway proxying to it
- SDK ai.chat() working end-to-end

**Execute:** `cd ~/lorai-workspace && claude` → "Read CLAUDE_PHASE2.md and implement all services."

---

## All Services Map

```
PORT 1842 GATEWAY
├── /v1/ (OpenAI-compatible)
│   ├── chat/completions ──→ Ollama         (Phase 1 ✅)
│   ├── completions ───────→ Ollama         (Phase 1 ✅)
│   ├── embeddings ────────→ Ollama         (Phase 1 ✅)
│   ├── models ────────────→ Ollama         (Phase 1 ✅)
│   ├── images/generations ─→ SDXL/FLUX     (Phase 2A)
│   ├── images/edits ──────→ SDXL img2img   (Phase 2A)
│   ├── audio/transcriptions→ Whisper.cpp   (Phase 2B)
│   └── audio/speech ──────→ Piper TTS      (Phase 2B)
├── /lorai/ (Native)
│   ├── hub/* ─────────────→               (Phase 1 ✅)
│   ├── video/generate ────→ CogVideoX     (Phase 2C)
│   ├── knowledge/ingest ──→ ChromaDB      (Phase 2D)
│   ├── knowledge/search ──→ ChromaDB      (Phase 2D)
│   ├── knowledge/ask ─────→ ChromaDB+LLM  (Phase 2D)
│   ├── agents/run ────────→ ReAct Loop    (Phase 2E)
│   ├── agents/list ───────→              (Phase 2E)
│   ├── agents/tools ──────→              (Phase 2E)
│   ├── code/execute ──────→ Sandbox       (Phase 2F)
│   ├── vision/analyze ────→ LLaVA/Ollama  (Phase 2G)
│   ├── lora/list ─────────→ /data/loras   (Phase 3A)
│   ├── lora/load ─────────→ Ollama        (Phase 3A)
│   ├── lora/unload ───────→ Ollama        (Phase 3A)
│   └── audio/music ───────→ MusicGen      (Phase 3B)
└── /api/health ───────────→               (Phase 1 ✅)
```

---

## All LLM Models Reference

| Service | Model | Size | Hardware | Install Command |
|---------|-------|------|----------|-----------------|
| Chat (default) | phi3:mini | 2.3GB | CPU | ollama pull phi3:mini |
| Chat (better) | llama3.2:8b | 4.7GB | CPU/GPU | ollama pull llama3.2 |
| Chat (best) | deepseek-v3 | 15GB+ | GPU 16GB+ | ollama pull deepseek-v3 |
| Code | qwen2.5-coder:7b | 4.4GB | CPU/GPU | ollama pull qwen2.5-coder:7b |
| Code (best) | qwen2.5-coder:32b | 20GB | GPU 24GB+ | ollama pull qwen2.5-coder:32b |
| Vision | llava:7b | 4.7GB | CPU/GPU | ollama pull llava:7b |
| Vision (better) | llama3.2-vision | 7.9GB | GPU 8GB+ | ollama pull llama3.2-vision |
| Embeddings | nomic-embed-text | 274MB | CPU | ollama pull nomic-embed-text |
| STT | whisper base.en | 142MB | CPU | Built-in whisper.cpp |
| STT (better) | whisper large-v3 | 3.1GB | GPU 4GB+ | Download GGML model |
| TTS | piper en_US-amy | 65MB | CPU | Built-in |
| Image | SDXL Turbo | 6.5GB | GPU 8GB+ | diffusers auto-download |
| Image (better) | FLUX.2 | 12GB | GPU 12GB+ | ComfyUI (Phase 3) |
| Video | CogVideoX-5B | 10GB | GPU 12GB+ | diffusers auto-download |
| Video (better) | Wan 2.2 | 28GB | GPU 24GB+ | Separate install |
| Music | MusicGen Small | 1.5GB | GPU 4GB+ | audiocraft library |

### Hardware Profiles

```python
# Auto-detect on gateway startup
GPU >= 24GB → "power"    → all services enabled
GPU >= 8GB  → "standard" → chat, code, image, vision, voice
GPU < 8GB   → "lite"     → chat, code, voice, embeddings only
No GPU      → "cpu"      → chat, code, voice, embeddings only
```

---

## PHASE 2A: Image Generation (GPU 8GB+)

### Dockerfile Additions

```dockerfile
# Image generation dependencies (loaded lazily on first request)
RUN pip3 install --break-system-packages \
    diffusers torch torchvision accelerate safetensors
```

### POST /v1/images/generations

OpenAI-compatible. Uses SDXL Turbo via diffusers (fast, 1-4 steps).

Request JSON:
```json
{
  "model": "sdxl-turbo",
  "prompt": "a sunset over mountains",
  "n": 1,
  "size": "1024x1024",
  "response_format": "url",
  "lorai_negative_prompt": "blurry, low quality",
  "lorai_steps": 4,
  "lorai_lora": ["watercolor-style"]
}
```

Response JSON:
```json
{
  "created": 1709000000,
  "data": [
    {"url": "http://localhost:1842/files/lorai_img_abc123.png"}
  ]
}
```

Implementation in gateway.py:
```python
_image_pipeline = None

def get_image_pipeline():
    global _image_pipeline
    if _image_pipeline is None:
        import torch
        if not torch.cuda.is_available():
            raise HTTPException(503, "Image generation requires GPU. Start with: lorai-workspace start --gpu")
        from diffusers import AutoPipelineForText2Image
        _image_pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
    return _image_pipeline

@app.post("/v1/images/generations")
async def generate_image(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    n = body.get("n", 1)
    size = body.get("size", "1024x1024")
    response_format = body.get("response_format", "url")
    negative = body.get("lorai_negative_prompt", "")
    steps = body.get("lorai_steps", 4)

    pipe = get_image_pipeline()
    w, h = map(int, size.split("x"))

    images = pipe(
        prompt=prompt, negative_prompt=negative,
        num_inference_steps=steps, width=w, height=h,
        num_images_per_prompt=n, guidance_scale=0.0
    ).images

    results = []
    for img in images:
        filename = f"lorai_img_{uuid4().hex[:8]}.png"
        filepath = f"/home/lorai/Desktop/{filename}"
        img.save(filepath)

        if response_format == "b64_json":
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="PNG")
            results.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
        else:
            results.append({"url": f"http://localhost:1842/files/{filename}"})

    return {"created": int(time.time()), "data": results}
```

### POST /v1/images/edits

Same pipeline but uses img2img mode. Takes input image + prompt.

```python
@app.post("/v1/images/edits")
async def edit_image(image: UploadFile, prompt: str = Form(...)):
    from PIL import Image as PILImage
    from diffusers import AutoPipelineForImage2Image
    img = PILImage.open(image.file).convert("RGB").resize((512, 512))

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16
    ).to("cuda")

    result = pipe(prompt=prompt, image=img, num_inference_steps=4,
                  guidance_scale=0.0, strength=0.5).images[0]

    filename = f"lorai_edit_{uuid4().hex[:8]}.png"
    filepath = f"/home/lorai/Desktop/{filename}"
    result.save(filepath)
    return {"created": int(time.time()), "data": [{"url": f"http://localhost:1842/files/{filename}"}]}
```

### Static File Serving (add to gateway)

```python
from fastapi.staticfiles import StaticFiles
app.mount("/files", StaticFiles(directory="/home/lorai/Desktop"), name="files")
```

---

## PHASE 2B: Voice (Whisper STT + Piper TTS)

### Dockerfile Additions

```dockerfile
# Whisper.cpp — Speech to Text
RUN apk add --no-cache ffmpeg && \
    git clone https://github.com/ggerganov/whisper.cpp /opt/whisper.cpp && \
    cd /opt/whisper.cpp && make -j$(nproc) && \
    bash models/download-ggml-model.sh base.en

# Piper TTS — Text to Speech
RUN pip3 install --break-system-packages piper-tts && \
    mkdir -p /data/voices && cd /data/voices && \
    curl -L -o en_US-amy-medium.onnx \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx" && \
    curl -L -o en_US-amy-medium.onnx.json \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
```

### POST /v1/audio/transcriptions

OpenAI-compatible Whisper endpoint.

Request: multipart/form-data with fields: file, model, language, response_format

```python
@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile, model: str = Form("whisper-1"), language: str = Form(None)):
    temp_path = f"/tmp/whisper_{uuid4().hex}{Path(file.filename).suffix}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Convert to 16kHz WAV
    wav_path = f"/tmp/whisper_{uuid4().hex}.wav"
    subprocess.run(["ffmpeg", "-i", temp_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
                   capture_output=True, timeout=30)

    # Run whisper.cpp
    cmd = ["/opt/whisper.cpp/main",
           "-m", "/opt/whisper.cpp/models/ggml-base.en.bin",
           "-f", wav_path, "--no-timestamps", "--print-progress", "false"]
    if language:
        cmd.extend(["-l", language])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    os.unlink(temp_path)
    os.unlink(wav_path)

    return {"text": result.stdout.strip()}
```

### POST /v1/audio/speech

OpenAI-compatible TTS endpoint.

Request JSON: model, input, voice, speed, response_format

```python
VOICE_MAP = {
    "nova": "/data/voices/en_US-amy-medium.onnx",
    "alloy": "/data/voices/en_US-amy-medium.onnx",
    "echo": "/data/voices/en_US-amy-medium.onnx",
    "fable": "/data/voices/en_US-amy-medium.onnx",
    "onyx": "/data/voices/en_US-amy-medium.onnx",
    "shimmer": "/data/voices/en_US-amy-medium.onnx",
}

@app.post("/v1/audio/speech")
async def text_to_speech(request: Request):
    body = await request.json()
    text = body.get("input", "")
    voice = body.get("voice", "nova")
    speed = body.get("speed", 1.0)

    voice_model = VOICE_MAP.get(voice, list(VOICE_MAP.values())[0])
    wav_path = f"/tmp/tts_{uuid4().hex}.wav"

    subprocess.run(
        ["piper", "--model", voice_model, "--output_file", wav_path,
         "--length-scale", str(1.0 / speed)],
        input=text, capture_output=True, text=True, timeout=30
    )

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(wav_path)

    return Response(content=audio_bytes, media_type="audio/wav")
```

---

## PHASE 2C: Video Generation (GPU 12GB+)

### Dockerfile Additions (optional, lazy-loaded)

```dockerfile
# Only needed if GPU >= 12GB
# RUN pip3 install --break-system-packages imageio[ffmpeg]
```

### POST /lorai/video/generate

Request JSON:
```json
{
  "prompt": "ocean waves crashing on rocks",
  "model": "cogvideo",
  "duration": 4,
  "fps": 8,
  "resolution": "720p",
  "image_base64": "optional base64 for img2vid"
}
```

```python
@app.post("/lorai/video/generate")
async def generate_video(request: Request):
    body = await request.json()
    import torch

    if not torch.cuda.is_available():
        raise HTTPException(503, "Video generation requires GPU. Start with: lorai-workspace start --gpu")

    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    if gpu_mem < 12:
        raise HTTPException(503, f"Video needs 12GB+ VRAM. You have {gpu_mem:.0f}GB.")

    prompt = body.get("prompt", "")
    duration = body.get("duration", 4)

    try:
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video

        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b", torch_dtype=torch.float16
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()

        video_frames = pipe(prompt=prompt, num_frames=duration * 8, guidance_scale=6.0).frames[0]

        filename = f"lorai_video_{uuid4().hex[:8]}.mp4"
        filepath = f"/home/lorai/Desktop/{filename}"
        export_to_video(video_frames, filepath, fps=8)

        return {
            "filepath": filepath,
            "url": f"http://localhost:1842/files/{filename}",
            "duration": duration,
            "model": body.get("model", "cogvideo"),
        }
    except ImportError:
        raise HTTPException(503, "Video dependencies not installed. Run: pip install diffusers[torch]")
```

---

## PHASE 2D: Knowledge / RAG

### Dockerfile Additions

Already have chromadb from Phase 1. Add:
```dockerfile
RUN pip3 install --break-system-packages sentence-transformers
```

Or rely on ChromaDB's built-in embedding (default: all-MiniLM-L6-v2).

### Create src/knowledge/knowledge_service.py

```python
import os
import hashlib
import chromadb
from pathlib import Path

VECTOR_DB_PATH = "/data/vectors"
SUPPORTED_EXTENSIONS = {".txt",".md",".py",".js",".ts",".json",".csv",".html",".xml",".yaml",".yml",".toml",".cfg",".ini",".sh",".sql",".r",".java",".go",".rs"}

class KnowledgeService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    def get_collection(self, name="default"):
        return self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def ingest(self, sources, collection="default", chunk_size=1000, chunk_overlap=200):
        coll = self.get_collection(collection)
        total_docs = 0
        total_chunks = 0

        for source in sources:
            source = os.path.expanduser(source)
            if os.path.isdir(source):
                files = [f for f in Path(source).rglob("*") if f.is_file()]
            elif os.path.isfile(source):
                files = [Path(source)]
            else:
                continue

            for filepath in files:
                if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    text = filepath.read_text(encoding="utf-8", errors="ignore")
                    if not text.strip():
                        continue
                    chunks = self._chunk(text, chunk_size, chunk_overlap)
                    file_id = hashlib.md5(str(filepath).encode()).hexdigest()

                    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": str(filepath), "chunk": i, "total_chunks": len(chunks)} for i in range(len(chunks))]

                    coll.upsert(ids=ids, documents=chunks, metadatas=metadatas)
                    total_docs += 1
                    total_chunks += len(chunks)
                except Exception:
                    continue

        return {"documents_ingested": total_docs, "chunks_created": total_chunks, "collection": collection}

    def search(self, query, collection="default", top_k=5, threshold=0.7):
        coll = self.get_collection(collection)
        if coll.count() == 0:
            return []

        results = coll.query(query_texts=[query], n_results=min(top_k, coll.count()))

        output = []
        for i in range(len(results["ids"][0])):
            score = 1.0 - (results["distances"][0][i] if results["distances"] else 1.0)
            if score >= threshold:
                output.append({
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i].get("source", ""),
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i],
                })
        return output

    def ask(self, question, collection="default", model="auto"):
        results = self.search(question, collection, top_k=5, threshold=0.5)
        if not results:
            return "No relevant documents found. Ingest documents first with ai.knowledge.ingest()."

        context = "\n\n---\n\n".join([f"Source: {r['source']}\n{r['content']}" for r in results[:5]])

        import httpx
        model_name = model if model != "auto" else os.getenv("LORAI_MODEL", "phi3:mini")
        resp = httpx.post("http://localhost:11434/v1/chat/completions", json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": f"Answer based on the context below. Cite sources when possible.\n\nContext:\n{context}"},
                {"role": "user", "content": question},
            ],
        }, timeout=60.0)
        return resp.json()["choices"][0]["message"]["content"]

    def _chunk(self, text, size=1000, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + size])
            start += size - overlap
        return chunks if chunks else [text]
```

### Gateway Routes

```python
knowledge_svc = KnowledgeService()

@app.post("/lorai/knowledge/ingest")
async def knowledge_ingest(request: Request):
    body = await request.json()
    return knowledge_svc.ingest(body.get("sources", []), body.get("collection", "default"),
                                body.get("chunk_size", 1000), body.get("chunk_overlap", 200))

@app.post("/lorai/knowledge/search")
async def knowledge_search(request: Request):
    body = await request.json()
    return {"results": knowledge_svc.search(body["query"], body.get("collection", "default"),
                                            body.get("top_k", 5), body.get("threshold", 0.7))}

@app.post("/lorai/knowledge/ask")
async def knowledge_ask(request: Request):
    body = await request.json()
    return {"answer": knowledge_svc.ask(body["question"], body.get("collection", "default"),
                                        body.get("model", "auto"))}
```

---

## PHASE 2E: Agents

### Simple ReAct Agent (no LangGraph dependency)

Create src/agents/agent_service.py:

```python
import json
import os
import subprocess
from pathlib import Path

class AgentService:
    TOOLS = {
        "search_files": {"desc": "Search files by name or content", "usage": "search term"},
        "read_file": {"desc": "Read file contents", "usage": "file path"},
        "write_file": {"desc": "Write to file. Line 1 = path, rest = content", "usage": "path\\ncontent"},
        "run_command": {"desc": "Execute shell command", "usage": "command string"},
        "knowledge_search": {"desc": "Semantic search in knowledge base", "usage": "search query"},
    }

    async def run(self, task, agents=None, tools=None, model="auto", max_steps=10, verbose=False):
        import httpx

        model_name = model if model != "auto" else os.getenv("LORAI_MODEL", "phi3:mini")
        steps = []
        tools_desc = json.dumps({k: v["desc"] for k, v in self.TOOLS.items()})

        messages = [
            {"role": "system", "content": f"""You are LorAI Agent. Solve tasks step by step.
Available tools: {tools_desc}

Respond with JSON on each step:
{{"thought": "reasoning", "action": "tool_name", "input": "tool input"}}

When done:
{{"thought": "reasoning", "action": "finish", "result": "final answer"}}"""},
            {"role": "user", "content": task},
        ]

        for step in range(max_steps):
            async with httpx.AsyncClient() as client:
                resp = await client.post("http://localhost:11434/v1/chat/completions",
                    json={"model": model_name, "messages": messages, "temperature": 0.2}, timeout=60.0)
            content = resp.json()["choices"][0]["message"]["content"]

            try:
                action = json.loads(content)
            except json.JSONDecodeError:
                action = {"thought": content, "action": "finish", "result": content}

            steps.append(action)

            if action.get("action") == "finish":
                return {
                    "result": action.get("result", content),
                    "steps": steps if verbose else [],
                    "agents_used": agents or ["default"],
                    "tools_used": list(set(s.get("action","") for s in steps if s.get("action") != "finish")),
                }

            tool_result = self._exec_tool(action.get("action", ""), action.get("input", ""))
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})

        return {"result": "Max steps reached. Partial results in steps.", "steps": steps if verbose else [],
                "agents_used": agents or ["default"], "tools_used": []}

    def _exec_tool(self, tool, inp):
        try:
            if tool == "run_command":
                r = subprocess.run(inp, shell=True, capture_output=True, text=True, timeout=30, cwd="/home/lorai")
                return (r.stdout + r.stderr)[:3000]
            elif tool == "read_file":
                return Path(os.path.expanduser(inp.strip())).read_text(errors="ignore")[:3000]
            elif tool == "write_file":
                lines = inp.split("\n", 1)
                if len(lines) == 2:
                    Path(lines[0].strip()).write_text(lines[1])
                    return f"Written to {lines[0].strip()}"
                return "Error: line 1 = path, rest = content"
            elif tool == "search_files":
                r = subprocess.run(f'find /home/lorai -iname "*{inp}*" 2>/dev/null | head -20',
                                   shell=True, capture_output=True, text=True, timeout=10)
                return r.stdout or "No files found"
            elif tool == "knowledge_search":
                from src.knowledge.knowledge_service import KnowledgeService
                results = KnowledgeService().search(inp, top_k=3, threshold=0.5)
                return json.dumps(results, indent=2)[:3000] if results else "No results"
            else:
                return f"Unknown tool: {tool}"
        except Exception as e:
            return f"Error: {str(e)}"
```

### Gateway Routes

```python
agent_svc = AgentService()

@app.post("/lorai/agents/run")
async def agents_run(request: Request):
    body = await request.json()
    return await agent_svc.run(body["task"], body.get("agents"), body.get("tools"),
                               body.get("model", "auto"), body.get("max_steps", 10), body.get("verbose", False))

@app.get("/lorai/agents/list")
async def agents_list():
    return {"agents": [
        {"name": "default", "description": "General-purpose agent with file and command tools"},
        {"name": "researcher", "description": "Web research agent (coming soon)"},
        {"name": "coder", "description": "Code generation and review agent (coming soon)"},
    ]}

@app.get("/lorai/agents/tools")
async def agents_tools():
    return {"tools": [{"name": k, "description": v["desc"]} for k, v in AgentService.TOOLS.items()]}
```

---

## PHASE 2F: Code Execution

### POST /lorai/code/execute

Sandboxed execution inside the container.

```python
@app.post("/lorai/code/execute")
async def execute_code(request: Request):
    body = await request.json()
    code = body.get("code", "")
    language = body.get("language", "python")
    timeout = min(body.get("timeout", 30), 60)

    cmd_map = {"python": ["python3", "-c"], "bash": ["bash", "-c"], "javascript": ["node", "-e"]}
    if language not in cmd_map:
        raise HTTPException(400, f"Unsupported: {language}. Use: python, bash, javascript")

    cmd = cmd_map[language] + [code]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                cwd="/home/lorai", env={**os.environ, "HOME": "/home/lorai"})
        return {"output": result.stdout, "error": result.stderr, "exit_code": result.returncode, "language": language}
    except subprocess.TimeoutExpired:
        return {"output": "", "error": f"Timed out ({timeout}s)", "exit_code": -1, "language": language}
```

Add `nodejs` to Dockerfile for JavaScript support.

---

## PHASE 2G: Vision

Vision uses Ollama multimodal models. No extra install — just pull llava.

### Setup (in entrypoint.sh or lazy on first request)

```bash
ollama pull llava:7b
```

### POST /lorai/vision/analyze

Convenience endpoint (could also use /v1/chat/completions with image content).

```python
@app.post("/lorai/vision/analyze")
async def vision_analyze(request: Request):
    body = await request.json()
    image_base64 = body.get("image_base64", "")
    prompt = body.get("prompt", "Describe this image in detail.")
    model = body.get("model", "llava:7b")

    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:11434/v1/chat/completions", json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }],
        }, timeout=60.0)
    return resp.json()
```

The SDK's `ai.vision.analyze()` and `ai.vision.ocr()` already encode the image to base64 and send it. This endpoint just proxies to Ollama's multimodal support.

---

## PHASE 3A: LoRA Management

Ollama supports LoRA via Modelfile ADAPTER directive.

### Gateway Routes

```python
@app.get("/lorai/lora/list")
async def lora_list():
    lora_dir = Path("/data/loras")
    lora_dir.mkdir(parents=True, exist_ok=True)
    loras = []
    for f in lora_dir.glob("*.gguf"):
        loras.append({"name": f.stem, "path": str(f), "size_mb": round(f.stat().st_size / 1e6, 1)})
    return {"loras": loras}

@app.post("/lorai/lora/load")
async def lora_load(request: Request):
    body = await request.json()
    name = body["name"]
    base_model = body.get("base_model", "llama3.2")

    lora_path = f"/data/loras/{name}.gguf"
    if not Path(lora_path).exists():
        raise HTTPException(404, f"LoRA not found: {name}. Place .gguf files in /data/loras/")

    model_name = f"{base_model}-{name}"
    modelfile = f"FROM {base_model}\nADAPTER {lora_path}"

    import httpx
    resp = httpx.post("http://localhost:11434/api/create",
                      json={"name": model_name, "modelfile": modelfile}, timeout=120.0)
    if resp.status_code != 200:
        raise HTTPException(500, f"Failed to create model: {resp.text}")

    return {"model": model_name, "base": base_model, "lora": name, "status": "loaded"}

@app.post("/lorai/lora/unload")
async def lora_unload(request: Request):
    body = await request.json()
    name = body["name"]
    import httpx
    resp = httpx.request("DELETE", "http://localhost:11434/api/delete",
                         json={"name": name}, timeout=30.0)
    return {"name": name, "status": "unloaded"}
```

Users place `.gguf` LoRA adapters in `~/.lorai/data/loras/` (mapped to `/data/loras/` inside container).

---

## PHASE 3B: Music Generation (GPU 4GB+)

### POST /lorai/audio/music

```python
@app.post("/lorai/audio/music")
async def generate_music(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "upbeat electronic music")
    duration = min(body.get("duration", 10), 30)

    try:
        import torch
        if not torch.cuda.is_available():
            raise HTTPException(503, "Music generation requires GPU.")

        from audiocraft.models import MusicGen
        import torchaudio

        model = MusicGen.get_pretrained("facebook/musicgen-small")
        model.set_generation_params(duration=duration)
        wav = model.generate([prompt])

        filename = f"lorai_music_{uuid4().hex[:8]}.wav"
        filepath = f"/home/lorai/Desktop/{filename}"
        torchaudio.save(filepath, wav[0].cpu(), sample_rate=32000)

        return {"filepath": filepath, "url": f"http://localhost:1842/files/{filename}",
                "duration": duration, "prompt": prompt}
    except ImportError:
        raise HTTPException(503, "MusicGen not installed. Run: pip install audiocraft")
```

---

## Testing All Services

After implementing everything, run these tests:

```bash
# ── Phase 1 (should already pass) ──
curl http://localhost:1842/api/health
curl http://localhost:1842/v1/models
curl -X POST http://localhost:1842/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"phi3:mini","messages":[{"role":"user","content":"Say hello"}]}'

# ── Phase 2B: Voice ──
# Create a test audio file first
curl -X POST http://localhost:1842/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from LorAI","voice":"nova"}' \
  --output /tmp/test_speech.wav
echo "TTS: $(file /tmp/test_speech.wav)"

curl -X POST http://localhost:1842/v1/audio/transcriptions \
  -F file=@/tmp/test_speech.wav -F model=whisper-1
echo "STT: should return 'Hello from LorAI'"

# ── Phase 2A: Image (GPU only) ──
curl -X POST http://localhost:1842/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a sunset","model":"sdxl-turbo","size":"512x512"}'

# ── Phase 2D: Knowledge ──
# Create test document
echo "LorAI runs on port 1842. It was inspired by Ada Lovelace." > /tmp/test_doc.txt
curl -X POST http://localhost:1842/lorai/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{"sources":["/tmp/test_doc.txt"],"collection":"test"}'

curl -X POST http://localhost:1842/lorai/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query":"what port","collection":"test"}'

curl -X POST http://localhost:1842/lorai/knowledge/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What port does LorAI use?","collection":"test"}'

# ── Phase 2E: Agents ──
curl -X POST http://localhost:1842/lorai/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task":"List all files on the desktop","verbose":true}'

curl http://localhost:1842/lorai/agents/list
curl http://localhost:1842/lorai/agents/tools

# ── Phase 2F: Code Execution ──
curl -X POST http://localhost:1842/lorai/code/execute \
  -H "Content-Type: application/json" \
  -d '{"code":"print(sum(range(100)))","language":"python"}'

curl -X POST http://localhost:1842/lorai/code/execute \
  -H "Content-Type: application/json" \
  -d '{"code":"echo Hello from bash","language":"bash"}'

# ── Phase 2G: Vision (needs llava model pulled) ──
# Encode a test image
IMG_B64=$(base64 -w0 /home/lorai/Desktop/some_image.png 2>/dev/null || echo "")
if [ -n "$IMG_B64" ]; then
  curl -X POST http://localhost:1842/lorai/vision/analyze \
    -H "Content-Type: application/json" \
    -d "{\"image_base64\":\"$IMG_B64\",\"prompt\":\"What is in this image?\"}"
fi

# ── Phase 3A: LoRA ──
curl http://localhost:1842/lorai/lora/list

# ── SDK Integration Test ──
python3 -c "
from lorai_workspace import LorAI
ai = LorAI(auto_start=False)

# Chat
print('Chat:', ai.chat('Say hello in 5 words'))

# Code
r = ai.code.generate('print fibonacci first 10', execute=True)
print('Code:', r.get('output',''))

# Knowledge
ai.knowledge.ingest(['/tmp/test_doc.txt'], collection='sdk_test')
print('Search:', ai.knowledge.search('port', collection='sdk_test'))
print('Ask:', ai.knowledge.ask('What port?', collection='sdk_test'))

# Hub
print('Models:', [m['name'] for m in ai.hub.models()])
print('Status:', ai.hub.status())

print('ALL SDK TESTS PASSED')
"
```

---

## Summary: What Claude Code Should Deliver

After executing both CLAUDE.md and CLAUDE_PHASE2.md:

| Service | Endpoint | Backend | Status |
|---------|----------|---------|--------|
| Chat | /v1/chat/completions | Ollama | Working |
| Completions | /v1/completions | Ollama | Working |
| Embeddings | /v1/embeddings | Ollama | Working |
| Models | /v1/models | Ollama | Working |
| Image Gen | /v1/images/generations | SDXL Turbo | GPU only |
| Image Edit | /v1/images/edits | SDXL img2img | GPU only |
| Voice STT | /v1/audio/transcriptions | Whisper.cpp | Working |
| Voice TTS | /v1/audio/speech | Piper TTS | Working |
| Video Gen | /lorai/video/generate | CogVideoX | GPU 12GB+ |
| Knowledge Ingest | /lorai/knowledge/ingest | ChromaDB | Working |
| Knowledge Search | /lorai/knowledge/search | ChromaDB | Working |
| Knowledge Ask | /lorai/knowledge/ask | ChromaDB+LLM | Working |
| Agents Run | /lorai/agents/run | ReAct+Ollama | Working |
| Agents List | /lorai/agents/list | Static | Working |
| Agents Tools | /lorai/agents/tools | Static | Working |
| Code Execute | /lorai/code/execute | subprocess | Working |
| Vision Analyze | /lorai/vision/analyze | LLaVA/Ollama | Working |
| LoRA List | /lorai/lora/list | /data/loras | Working |
| LoRA Load | /lorai/lora/load | Ollama | Working |
| LoRA Unload | /lorai/lora/unload | Ollama | Working |
| Music Gen | /lorai/audio/music | MusicGen | GPU 4GB+ |
| Health | /api/health | Gateway | Working |
| Hub Status | /lorai/hub/status | Gateway | Working |
| Hub Models | /lorai/hub/models | Ollama | Working |
| Hub Pull | /lorai/hub/pull | Ollama | Working |
| Hub Remove | /lorai/hub/remove | Ollama | Working |
| Hub Bench | /lorai/hub/bench | Gateway | Working |
| Static Files | /files/* | Desktop dir | Working |

**Total: 27 endpoints. All specified. All testable.**
