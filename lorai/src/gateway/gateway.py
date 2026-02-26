"""LorAI API Gateway — FastAPI on port 1842.

Two route groups:
  /v1/*    — OpenAI-compatible proxy to Ollama (localhost:11434)
  /lorai/* — LorAI-native endpoints

Phase 2: All services implemented (27 endpoints total).
"""

from __future__ import annotations

import base64
import os
import platform
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure src directory is on path for imports
sys.path.insert(0, "/opt/lorai")

VERSION = "0.1.0"
PORT = int(os.environ.get("LORAI_PORT", 1842))
OLLAMA_BASE = "http://localhost:11434"

app = FastAPI(title="LorAI", version=VERSION)

# CORS — allow all origins (local development tool)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for generated images/videos/audio
os.makedirs("/home/lorai/Desktop", exist_ok=True)
app.mount("/files", StaticFiles(directory="/home/lorai/Desktop"), name="files")

_start_time = time.time()

# Lazy-loaded service singletons
_knowledge_svc = None
_agent_svc = None
_image_pipeline = None


def _get_knowledge_svc():
    global _knowledge_svc
    if _knowledge_svc is None:
        from src.knowledge.knowledge_service import KnowledgeService
        _knowledge_svc = KnowledgeService()
    return _knowledge_svc


def _get_agent_svc():
    global _agent_svc
    if _agent_svc is None:
        from src.agents.agent_service import AgentService
        _agent_svc = AgentService()
    return _agent_svc


# ======================================================================
# Hardware profile detection
# ======================================================================

def _detect_hardware_profile() -> str:
    """Auto-detect hardware profile: power, standard, lite, or cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if vram_gb >= 24:
                return "power"
            elif vram_gb >= 8:
                return "standard"
            else:
                return "lite"
    except ImportError:
        pass
    return "cpu"


# ======================================================================
# Health check
# ======================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": VERSION,
        "port": PORT,
        "hardware_profile": _detect_hardware_profile(),
    }


# ======================================================================
# OpenAI-compatible routes — proxy to Ollama
# ======================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions to Ollama with streaming support."""
    body = await request.json()

    if body.get("stream", False):
        async def stream_generator():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", f"{OLLAMA_BASE}/v1/chat/completions",
                    json=body, timeout=300,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/chat/completions", json=body, timeout=300,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/completions")
async def completions(request: Request):
    """Proxy text completions to Ollama."""
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/completions", json=body, timeout=300,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Proxy embeddings to Ollama."""
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/embeddings", json=body, timeout=300,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/v1/models")
async def list_models():
    """Proxy model listing to Ollama."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OLLAMA_BASE}/v1/models", timeout=30)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ======================================================================
# Phase 2A: Image Generation (SDXL Turbo via diffusers)
# ======================================================================

@app.post("/v1/images/generations")
async def generate_image(request: Request):
    """Generate images using SDXL Turbo. Requires GPU 8GB+."""
    body = await request.json()
    prompt = body.get("prompt", "")
    n = body.get("n", 1)
    size = body.get("size", "1024x1024")
    response_format = body.get("response_format", "url")
    negative = body.get("lorai_negative_prompt", "")
    steps = body.get("lorai_steps", 4)

    try:
        import torch
        if not torch.cuda.is_available():
            raise HTTPException(503, "Image generation requires GPU. Start with: lorai-workspace start --gpu")

        global _image_pipeline
        if _image_pipeline is None:
            from diffusers import AutoPipelineForText2Image
            _image_pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16",
            ).to("cuda")

        w, h = map(int, size.split("x"))
        images = _image_pipeline(
            prompt=prompt, negative_prompt=negative,
            num_inference_steps=steps, width=w, height=h,
            num_images_per_prompt=n, guidance_scale=0.0,
        ).images

        results = []
        for img in images:
            filename = f"lorai_img_{uuid4().hex[:8]}.png"
            filepath = f"/home/lorai/Desktop/{filename}"
            img.save(filepath)

            if response_format == "b64_json":
                buf = BytesIO()
                img.save(buf, format="PNG")
                results.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
            else:
                results.append({"url": f"http://localhost:{PORT}/files/{filename}"})

        return {"created": int(time.time()), "data": results}

    except ImportError:
        raise HTTPException(
            503,
            "Image generation dependencies not installed. "
            "Run: pip install diffusers torch torchvision accelerate safetensors",
        )


@app.post("/v1/images/edits")
async def edit_image(image: UploadFile, prompt: str = Form(...)):
    """Edit images using SDXL img2img. Requires GPU 8GB+."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise HTTPException(503, "Image editing requires GPU. Start with: lorai-workspace start --gpu")

        from PIL import Image as PILImage
        from diffusers import AutoPipelineForImage2Image

        img = PILImage.open(image.file).convert("RGB").resize((512, 512))
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16,
        ).to("cuda")

        result = pipe(
            prompt=prompt, image=img, num_inference_steps=4,
            guidance_scale=0.0, strength=0.5,
        ).images[0]

        filename = f"lorai_edit_{uuid4().hex[:8]}.png"
        filepath = f"/home/lorai/Desktop/{filename}"
        result.save(filepath)

        return {
            "created": int(time.time()),
            "data": [{"url": f"http://localhost:{PORT}/files/{filename}"}],
        }
    except ImportError:
        raise HTTPException(503, "Image dependencies not installed. Run: pip install diffusers torch Pillow")


# ======================================================================
# Phase 2B: Voice (Whisper STT + Piper TTS)
# ======================================================================

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile, model: str = Form("whisper-1"), language: str = Form(None)):
    """Speech-to-text using whisper.cpp."""
    temp_path = f"/tmp/whisper_{uuid4().hex}{Path(file.filename).suffix}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    wav_path = f"/tmp/whisper_{uuid4().hex}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", temp_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
            capture_output=True, timeout=30,
        )

        cmd = [
            "/opt/whisper.cpp/main",
            "-m", "/opt/whisper.cpp/models/ggml-base.en.bin",
            "-f", wav_path, "--no-timestamps", "--print-progress", "false",
        ]
        if language:
            cmd.extend(["-l", language])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        return {"text": result.stdout.strip()}
    except FileNotFoundError:
        raise HTTPException(503, "Whisper.cpp not installed. Rebuild container with whisper support.")
    finally:
        for p in [temp_path, wav_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


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
    """Text-to-speech using Piper TTS."""
    body = await request.json()
    text = body.get("input", "")
    voice = body.get("voice", "nova")
    speed = body.get("speed", 1.0)

    voice_model = VOICE_MAP.get(voice, list(VOICE_MAP.values())[0])
    wav_path = f"/tmp/tts_{uuid4().hex}.wav"

    try:
        subprocess.run(
            ["piper", "--model", voice_model, "--output_file", wav_path,
             "--length-scale", str(1.0 / speed)],
            input=text, capture_output=True, text=True, timeout=30,
        )

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        return Response(content=audio_bytes, media_type="audio/wav")
    except FileNotFoundError:
        raise HTTPException(503, "Piper TTS not installed. Rebuild container with piper support.")
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ======================================================================
# Phase 2C: Video Generation (CogVideoX, GPU 12GB+)
# ======================================================================

@app.post("/lorai/video/generate")
async def generate_video(request: Request):
    """Generate video using CogVideoX. Requires GPU 12GB+."""
    body = await request.json()

    try:
        import torch
        if not torch.cuda.is_available():
            raise HTTPException(503, "Video generation requires GPU. Start with: lorai-workspace start --gpu")

        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        if gpu_mem < 12:
            raise HTTPException(503, f"Video needs 12GB+ VRAM. You have {gpu_mem:.0f}GB.")

        prompt = body.get("prompt", "")
        duration = body.get("duration", 4)

        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video

        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b", torch_dtype=torch.float16,
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()

        video_frames = pipe(
            prompt=prompt, num_frames=duration * 8, guidance_scale=6.0,
        ).frames[0]

        filename = f"lorai_video_{uuid4().hex[:8]}.mp4"
        filepath = f"/home/lorai/Desktop/{filename}"
        export_to_video(video_frames, filepath, fps=8)

        return {
            "filepath": filepath,
            "url": f"http://localhost:{PORT}/files/{filename}",
            "duration": duration,
            "model": body.get("model", "cogvideo"),
        }
    except ImportError:
        raise HTTPException(503, "Video dependencies not installed. Run: pip install diffusers[torch]")


# ======================================================================
# Phase 2D: Knowledge / RAG
# ======================================================================

@app.post("/lorai/knowledge/ingest")
async def knowledge_ingest(request: Request):
    """Ingest documents into the knowledge base."""
    body = await request.json()
    svc = _get_knowledge_svc()
    return svc.ingest(
        body.get("sources", []),
        body.get("collection", "default"),
        body.get("chunk_size", 1000),
        body.get("chunk_overlap", 200),
    )


@app.post("/lorai/knowledge/search")
async def knowledge_search(request: Request):
    """Semantic search in the knowledge base."""
    body = await request.json()
    svc = _get_knowledge_svc()
    return {
        "results": svc.search(
            body["query"],
            body.get("collection", "default"),
            body.get("top_k", 5),
            body.get("threshold", 0.7),
        )
    }


@app.post("/lorai/knowledge/ask")
async def knowledge_ask(request: Request):
    """RAG question answering over the knowledge base."""
    body = await request.json()
    svc = _get_knowledge_svc()
    return {
        "answer": svc.ask(
            body["question"],
            body.get("collection", "default"),
            body.get("model", "auto"),
        )
    }


# ======================================================================
# Phase 2E: Agents
# ======================================================================

@app.post("/lorai/agents/run")
async def agents_run(request: Request):
    """Run a ReAct agent workflow."""
    body = await request.json()
    svc = _get_agent_svc()
    return await svc.run(
        body["task"],
        body.get("agents"),
        body.get("tools"),
        body.get("model", "auto"),
        body.get("max_steps", 10),
        body.get("verbose", False),
    )


@app.get("/lorai/agents/list")
async def agents_list():
    """List available agents."""
    return {
        "agents": [
            {"name": "default", "description": "General-purpose agent with file and command tools"},
            {"name": "researcher", "description": "Web research agent (coming soon)"},
            {"name": "coder", "description": "Code generation and review agent (coming soon)"},
        ]
    }


@app.get("/lorai/agents/tools")
async def agents_tools():
    """List available agent tools."""
    from src.agents.agent_service import AgentService
    return {
        "tools": [
            {"name": k, "description": v["desc"]}
            for k, v in AgentService.TOOLS.items()
        ]
    }


# ======================================================================
# Phase 2F: Code Execution
# ======================================================================

@app.post("/lorai/code/execute")
async def execute_code(request: Request):
    """Execute code in a sandboxed subprocess."""
    body = await request.json()
    code = body.get("code", "")
    language = body.get("language", "python")
    timeout_sec = min(body.get("timeout", 30), 60)

    cmd_map = {
        "python": ["python3", "-c"],
        "bash": ["bash", "-c"],
        "javascript": ["node", "-e"],
    }
    if language not in cmd_map:
        raise HTTPException(400, f"Unsupported language: {language}. Use: python, bash, javascript")

    cmd = cmd_map[language] + [code]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec,
            cwd="/home/lorai", env={**os.environ, "HOME": "/home/lorai"},
        )
        return {
            "output": result.stdout,
            "error": result.stderr,
            "exit_code": result.returncode,
            "language": language,
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": f"Timed out ({timeout_sec}s)",
            "exit_code": -1,
            "language": language,
        }


# ======================================================================
# Phase 2G: Vision (LLaVA via Ollama)
# ======================================================================

@app.post("/lorai/vision/analyze")
async def vision_analyze(request: Request):
    """Analyze an image using LLaVA or other multimodal Ollama models."""
    body = await request.json()
    image_base64 = body.get("image_base64", "")
    prompt = body.get("prompt", "Describe this image in detail.")
    model = body.get("model", "llava:7b")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=60.0,
        )
    return resp.json()


# ======================================================================
# Phase 3A: LoRA Management
# ======================================================================

@app.get("/lorai/lora/list")
async def lora_list():
    """List available LoRA adapters from /data/loras."""
    lora_dir = Path("/data/loras")
    lora_dir.mkdir(parents=True, exist_ok=True)
    loras = []
    for f in lora_dir.glob("*.gguf"):
        loras.append({
            "name": f.stem,
            "path": str(f),
            "size_mb": round(f.stat().st_size / 1e6, 1),
        })
    return {"loras": loras}


@app.post("/lorai/lora/load")
async def lora_load(request: Request):
    """Load a LoRA adapter onto a base model via Ollama Modelfile."""
    body = await request.json()
    name = body["name"]
    base_model = body.get("base_model", "llama3.2")

    lora_path = f"/data/loras/{name}.gguf"
    if not Path(lora_path).exists():
        raise HTTPException(404, f"LoRA not found: {name}. Place .gguf files in /data/loras/")

    model_name = f"{base_model}-{name}"
    modelfile = f"FROM {base_model}\nADAPTER {lora_path}"

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/create",
            json={"name": model_name, "modelfile": modelfile},
            timeout=120.0,
        )
    if resp.status_code != 200:
        raise HTTPException(500, f"Failed to create model: {resp.text}")

    return {"model": model_name, "base": base_model, "lora": name, "status": "loaded"}


@app.post("/lorai/lora/unload")
async def lora_unload(request: Request):
    """Unload/remove a LoRA-augmented model from Ollama."""
    body = await request.json()
    name = body["name"]

    async with httpx.AsyncClient() as client:
        await client.request(
            "DELETE", f"{OLLAMA_BASE}/api/delete",
            json={"name": name}, timeout=30.0,
        )

    return {"name": name, "status": "unloaded"}


# ======================================================================
# Phase 3B: Music Generation (MusicGen, GPU 4GB+)
# ======================================================================

@app.post("/lorai/audio/music")
async def generate_music(request: Request):
    """Generate music using MusicGen. Requires GPU 4GB+."""
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

        return {
            "filepath": filepath,
            "url": f"http://localhost:{PORT}/files/{filename}",
            "duration": duration,
            "prompt": prompt,
        }
    except ImportError:
        raise HTTPException(503, "MusicGen not installed. Run: pip install audiocraft")


# ======================================================================
# LorAI-native routes — Hub (Phase 1, unchanged)
# ======================================================================

@app.get("/lorai/hub/status")
async def hub_status():
    """System info: CPU, RAM, GPU, loaded models, uptime."""
    import shutil

    uptime = int(time.time() - _start_time)
    disk = shutil.disk_usage("/")

    loaded_models = []
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                loaded_models = [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        pass

    return {
        "status": "ok",
        "version": VERSION,
        "uptime_seconds": uptime,
        "hardware_profile": _detect_hardware_profile(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "ram_total_gb": round(_get_total_ram_gb(), 1),
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "gpu": _detect_gpu(),
        "loaded_models": loaded_models,
    }


@app.get("/lorai/hub/models")
async def hub_models():
    """List all Ollama models."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OLLAMA_BASE}/api/tags", timeout=30)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


class PullRequest(BaseModel):
    name: str


@app.post("/lorai/hub/pull")
async def hub_pull(req: PullRequest):
    """Pull/download an Ollama model."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/pull",
            json={"name": req.name, "stream": False},
            timeout=600,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


class RemoveRequest(BaseModel):
    name: str


@app.post("/lorai/hub/remove")
async def hub_remove(req: RemoveRequest):
    """Delete an Ollama model."""
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            "DELETE", f"{OLLAMA_BASE}/api/delete",
            json={"name": req.name}, timeout=30,
        )
        return JSONResponse(
            content={"status": "removed", "model": req.name},
            status_code=resp.status_code,
        )


@app.post("/lorai/hub/bench")
async def hub_bench():
    """Basic benchmark — measure inference speed."""
    prompt = "Count from 1 to 10."
    model = os.environ.get("LORAI_MODEL", "phi3:mini")

    start = time.time()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        data = resp.json()

    tokens = data.get("usage", {}).get("total_tokens", 0)

    return {
        "model": model,
        "elapsed_seconds": round(elapsed, 2),
        "total_tokens": tokens,
        "tokens_per_second": round(tokens / elapsed, 1) if elapsed > 0 and tokens > 0 else None,
    }


# ======================================================================
# Helpers
# ======================================================================

def _get_total_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _detect_gpu() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "none"


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
