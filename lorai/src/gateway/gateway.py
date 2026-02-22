"""LorAI API Gateway — FastAPI on port 1842.

Two route groups:
  /v1/*    — OpenAI-compatible proxy to Ollama (localhost:11434)
  /lorai/* — LorAI-native endpoints
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

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

_start_time = time.time()


# ======================================================================
# Health check
# ======================================================================

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": VERSION, "port": PORT}


# ======================================================================
# OpenAI-compatible routes — proxy to Ollama
# ======================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions to Ollama with streaming support."""
    body = await request.json()

    # Streaming
    if body.get("stream", False):
        async def stream_generator():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE}/v1/chat/completions",
                    json=body,
                    timeout=300,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/chat/completions",
            json=body,
            timeout=300,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/completions")
async def completions(request: Request):
    """Proxy text completions to Ollama."""
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/completions",
            json=body,
            timeout=300,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Proxy embeddings to Ollama."""
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/embeddings",
            json=body,
            timeout=300,
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/v1/models")
async def list_models():
    """Proxy model listing to Ollama."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OLLAMA_BASE}/v1/models", timeout=30)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/images/generations")
async def image_generations():
    """Image generation — not yet available."""
    return JSONResponse(
        status_code=501,
        content={"error": {"message": "Image generation not installed. Run: lorai install comfyui", "type": "not_implemented"}},
    )


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions():
    """Audio transcription — not yet available."""
    return JSONResponse(
        status_code=501,
        content={"error": {"message": "Speech-to-text not installed. Run: lorai install whisper", "type": "not_implemented"}},
    )


@app.post("/v1/audio/speech")
async def audio_speech():
    """Text-to-speech — not yet available."""
    return JSONResponse(
        status_code=501,
        content={"error": {"message": "Text-to-speech not installed. Run: lorai install piper", "type": "not_implemented"}},
    )


# ======================================================================
# LorAI-native routes — Hub (Phase 1)
# ======================================================================

@app.get("/lorai/hub/status")
async def hub_status():
    """System info: CPU, RAM, GPU, loaded models, uptime."""
    import shutil

    uptime = int(time.time() - _start_time)
    disk = shutil.disk_usage("/")

    # Try to get loaded models from Ollama
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
            "DELETE",
            f"{OLLAMA_BASE}/api/delete",
            json={"name": req.name},
            timeout=30,
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
# LorAI-native routes — Phase 2+ stubs (501)
# ======================================================================

@app.post("/lorai/video/generate")
async def video_generate():
    return JSONResponse(status_code=501, content={"error": "Video generation coming in Phase 2"})


@app.post("/lorai/knowledge/ingest")
async def knowledge_ingest():
    return JSONResponse(status_code=501, content={"error": "Knowledge base coming in Phase 2"})


@app.post("/lorai/knowledge/search")
async def knowledge_search():
    return JSONResponse(status_code=501, content={"error": "Knowledge base coming in Phase 2"})


@app.post("/lorai/knowledge/ask")
async def knowledge_ask():
    return JSONResponse(status_code=501, content={"error": "Knowledge base coming in Phase 2"})


@app.post("/lorai/agents/run")
async def agents_run():
    return JSONResponse(status_code=501, content={"error": "Agent workflows coming in Phase 2"})


@app.get("/lorai/agents/list")
async def agents_list():
    return JSONResponse(status_code=501, content={"error": "Agent workflows coming in Phase 2"})


@app.get("/lorai/agents/tools")
async def agents_tools():
    return JSONResponse(status_code=501, content={"error": "Agent workflows coming in Phase 2"})


@app.get("/lorai/lora/list")
async def lora_list():
    return JSONResponse(status_code=501, content={"error": "LoRA management coming in Phase 3"})


@app.post("/lorai/lora/load")
async def lora_load():
    return JSONResponse(status_code=501, content={"error": "LoRA management coming in Phase 3"})


@app.post("/lorai/lora/unload")
async def lora_unload():
    return JSONResponse(status_code=501, content={"error": "LoRA management coming in Phase 3"})


@app.post("/lorai/code/execute")
async def code_execute():
    return JSONResponse(status_code=501, content={"error": "Code execution coming in Phase 2"})


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
        import subprocess
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
