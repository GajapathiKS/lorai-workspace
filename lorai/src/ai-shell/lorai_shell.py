"""LorAI Shell — Natural language to system commands.

FastAPI backend that takes natural language input, queries Ollama,
and returns structured actions (commands to execute, files to create, etc.).
"""

from __future__ import annotations

import os

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LorAI Shell")

OLLAMA_BASE = "http://localhost:11434"
MODEL = os.environ.get("LORAI_MODEL", "phi3:mini")

SYSTEM_PROMPT = """You are LorAI Shell, an AI assistant running inside a Linux desktop environment.
When the user gives you a natural language instruction, respond with a JSON object:

{
  "thought": "brief explanation of what you'll do",
  "actions": [
    {"type": "command", "value": "the shell command to run"},
    {"type": "say", "value": "what to say to the user"}
  ]
}

Available action types: command (run a shell command), say (respond to user),
file_create (create a file), file_edit (edit a file), browse (open URL).

Keep commands safe. Never run destructive commands without the user asking.
Always explain what you're doing."""


class ShellRequest(BaseModel):
    message: str
    context: list[dict] | None = None


class ShellResponse(BaseModel):
    thought: str
    actions: list[dict]
    raw: str


@app.post("/shell/ask")
async def shell_ask(req: ShellRequest):
    """Process a natural language request and return structured actions."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if req.context:
        messages.extend(req.context)

    messages.append({"role": "user", "content": req.message})

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

    import json
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"thought": "I understood your request.", "actions": [{"type": "say", "value": content}]}

    return ShellResponse(
        thought=parsed.get("thought", ""),
        actions=parsed.get("actions", []),
        raw=content,
    )


@app.get("/shell/health")
async def shell_health():
    return {"status": "ok", "service": "lorai-shell"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8421, log_level="warning")
