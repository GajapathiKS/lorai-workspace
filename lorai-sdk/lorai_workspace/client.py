"""LorAI client — extends OpenAI with 10 AI services.

Usage:
    from lorai_workspace import LorAI
    ai = LorAI()
    print(ai.chat("Hello!"))
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from lorai_workspace.docker import ensure_running, stop_container


class LorAI(OpenAI):
    """OpenAI-compatible client backed by a local LorAI Docker container.

    Extends the official OpenAI Python SDK so that any existing OpenAI code
    works unchanged — just swap ``OpenAI()`` for ``LorAI()``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        port: int = 1842,
        auto_start: bool = True,
        gpu: bool = False,
        default_model: str = "auto",
    ) -> None:
        self._port = port
        self._default_model = default_model

        if auto_start:
            ensure_running(port=port, gpu=gpu)

        super().__init__(
            base_url=base_url or f"http://localhost:{port}/v1",
            api_key=api_key or "not-needed",
        )

        # Initialize service objects
        self.image = _ImageService(self)
        self.video = _VideoService(self)
        self.voice = _VoiceService(self)
        self.knowledge = _KnowledgeService(self)
        self.agents = _AgentsService(self)
        self.code = _CodeService(self)
        self.vision = _VisionService(self)
        self.lora = _LoRAService(self)
        self.hub = _HubService(self)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        *,
        model: str | None = None,
        system: str | None = None,
        lora: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        json_mode: bool = False,
    ) -> str:
        """Simple one-shot chat. Returns the assistant's text reply."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        kwargs: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Attach LoRA header if requested
        extra_headers = {}
        if lora:
            extra_headers["X-LorAI-LoRA"] = lora

        if stream:
            chunks = []
            response = self.chat.completions.create(
                **kwargs, stream=True, extra_headers=extra_headers or None,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    chunks.append(delta)
            return "".join(chunks)

        response = self.chat.completions.create(
            **kwargs, extra_headers=extra_headers or None,
        )
        return response.choices[0].message.content

    def stop(self) -> None:
        """Stop the LorAI Docker container."""
        stop_container()

    @property
    def _native_base(self) -> str:
        return f"http://localhost:{self._port}"


# ======================================================================
# Service classes
# ======================================================================

class _BaseNativeService:
    """Base for services that call LorAI-native endpoints."""

    def __init__(self, client: LorAI) -> None:
        self._client = client

    def _post(self, path: str, **kwargs: Any) -> Any:
        resp = httpx.post(
            f"{self._client._native_base}{path}",
            json=kwargs, timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> Any:
        resp = httpx.get(
            f"{self._client._native_base}{path}",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


class _ImageService:
    """Image generation via OpenAI /v1/images endpoints."""

    def __init__(self, client: LorAI) -> None:
        self._client = client

    def generate(
        self,
        prompt: str,
        *,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        lora: str | None = None,
        save_to: str | None = None,
    ) -> Any:
        extra_headers = {}
        if lora:
            extra_headers["X-LorAI-LoRA"] = lora
        response = self._client.images.generate(
            model=model, prompt=prompt, size=size,
            extra_headers=extra_headers or None,
        )
        if save_to and response.data and response.data[0].b64_json:
            Path(save_to).write_bytes(base64.b64decode(response.data[0].b64_json))
        return response

    def edit(self, image_path: str, prompt: str, **kwargs: Any) -> Any:
        with open(image_path, "rb") as f:
            return self._client.images.edit(image=f, prompt=prompt, **kwargs)


class _VideoService(_BaseNativeService):
    """Video generation via LorAI-native /lorai/video endpoints."""

    def generate(
        self,
        prompt: str,
        *,
        model: str = "auto",
        duration: float = 4.0,
        fps: int = 24,
        image_path: str | None = None,
        save_to: str | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "prompt": prompt, "model": model,
            "duration": duration, "fps": fps,
        }
        if image_path:
            payload["image"] = base64.b64encode(Path(image_path).read_bytes()).decode()
        return self._post("/lorai/video/generate", **payload)


class _VoiceService:
    """Voice services via OpenAI /v1/audio endpoints."""

    def __init__(self, client: LorAI) -> None:
        self._client = client

    def transcribe(self, audio_path: str, *, model: str = "whisper-1") -> str:
        with open(audio_path, "rb") as f:
            result = self._client.audio.transcriptions.create(model=model, file=f)
        return result.text

    def speak(
        self, text: str, *, voice: str = "alloy", save_to: str | None = None,
    ) -> Any:
        response = self._client.audio.speech.create(
            model="tts-1", voice=voice, input=text,
        )
        if save_to:
            response.write_to_file(save_to)
        return response


class _KnowledgeService(_BaseNativeService):
    """RAG / knowledge base via LorAI-native /lorai/knowledge endpoints."""

    def ingest(self, source: str, *, collection: str = "default") -> Any:
        return self._post("/lorai/knowledge/ingest", source=source, collection=collection)

    def search(self, query: str, *, top_k: int = 5) -> Any:
        return self._post("/lorai/knowledge/search", query=query, top_k=top_k)

    def ask(self, question: str) -> Any:
        return self._post("/lorai/knowledge/ask", question=question)


class _AgentsService(_BaseNativeService):
    """Agent workflows via LorAI-native /lorai/agents endpoints."""

    def run(
        self,
        task: str,
        *,
        agents: list[str] | None = None,
        tools: list[str] | None = None,
        max_steps: int = 10,
    ) -> Any:
        return self._post(
            "/lorai/agents/run",
            task=task, agents=agents or [], tools=tools or [],
            max_steps=max_steps,
        )

    def list_agents(self) -> Any:
        return self._get("/lorai/agents/list")

    def list_tools(self) -> Any:
        return self._get("/lorai/agents/tools")


class _CodeService(_BaseNativeService):
    """Code generation and execution."""

    def __init__(self, client: LorAI) -> None:
        super().__init__(client)

    def generate(
        self, prompt: str, *, language: str = "python", execute: bool = False,
    ) -> Any:
        return self._post(
            "/lorai/code/execute",
            prompt=prompt, language=language, execute=execute,
        )

    def review(self, code: str) -> str:
        response = self._client.chat.completions.create(
            model=self._client._default_model,
            messages=[
                {"role": "system", "content": "You are a code reviewer. Review the following code and provide feedback."},
                {"role": "user", "content": code},
            ],
        )
        return response.choices[0].message.content


class _VisionService:
    """Vision analysis via OpenAI multimodal endpoints."""

    def __init__(self, client: LorAI) -> None:
        self._client = client

    def analyze(self, image_path: str, prompt: str = "Describe this image.") -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        response = self._client.chat.completions.create(
            model=self._client._default_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        )
        return response.choices[0].message.content

    def ocr(self, image_path: str) -> str:
        return self.analyze(image_path, prompt="Extract all text from this image (OCR).")


class _LoRAService(_BaseNativeService):
    """LoRA adapter management via /lorai/lora endpoints."""

    def list(self) -> Any:
        return self._get("/lorai/lora/list")

    def load(self, name: str, *, base_model: str = "auto") -> Any:
        return self._post("/lorai/lora/load", name=name, base_model=base_model)

    def unload(self, name: str) -> Any:
        return self._post("/lorai/lora/unload", name=name)


class _HubService(_BaseNativeService):
    """Model hub management via /lorai/hub endpoints."""

    def models(self) -> Any:
        return self._get("/lorai/hub/models")

    def pull(self, model: str) -> Any:
        return self._post("/lorai/hub/pull", name=model)

    def remove(self, model: str) -> Any:
        return self._post("/lorai/hub/remove", name=model)

    def status(self) -> Any:
        return self._get("/lorai/hub/status")

    def bench(self) -> Any:
        return self._post("/lorai/hub/bench")
