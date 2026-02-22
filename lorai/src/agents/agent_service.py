"""LorAI Agent Service — Simple ReAct agent loop.

Uses Ollama for reasoning and provides tool execution
for file operations, commands, and knowledge search.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


class AgentService:
    TOOLS = {
        "search_files": {
            "desc": "Search files by name or content",
            "usage": "search term",
        },
        "read_file": {
            "desc": "Read file contents",
            "usage": "file path",
        },
        "write_file": {
            "desc": "Write to file. Line 1 = path, rest = content",
            "usage": "path\\ncontent",
        },
        "run_command": {
            "desc": "Execute shell command",
            "usage": "command string",
        },
        "knowledge_search": {
            "desc": "Semantic search in knowledge base",
            "usage": "search query",
        },
    }

    async def run(
        self,
        task: str,
        agents: list[str] | None = None,
        tools: list[str] | None = None,
        model: str = "auto",
        max_steps: int = 10,
        verbose: bool = False,
    ) -> dict:
        import httpx

        model_name = (
            model if model != "auto" else os.getenv("LORAI_MODEL", "phi3:mini")
        )
        steps = []
        tools_desc = json.dumps({k: v["desc"] for k, v in self.TOOLS.items()})

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are LorAI Agent. Solve tasks step by step.\n"
                    f"Available tools: {tools_desc}\n\n"
                    f"Respond with JSON on each step:\n"
                    f'{{"thought": "reasoning", "action": "tool_name", "input": "tool input"}}\n\n'
                    f"When done:\n"
                    f'{{"thought": "reasoning", "action": "finish", "result": "final answer"}}'
                ),
            },
            {"role": "user", "content": task},
        ]

        for step in range(max_steps):
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://localhost:11434/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.2,
                    },
                    timeout=60.0,
                )
            content = resp.json()["choices"][0]["message"]["content"]

            try:
                action = json.loads(content)
            except json.JSONDecodeError:
                action = {
                    "thought": content,
                    "action": "finish",
                    "result": content,
                }

            steps.append(action)

            if action.get("action") == "finish":
                return {
                    "result": action.get("result", content),
                    "steps": steps if verbose else [],
                    "agents_used": agents or ["default"],
                    "tools_used": list(
                        set(
                            s.get("action", "")
                            for s in steps
                            if s.get("action") != "finish"
                        )
                    ),
                }

            tool_result = self._exec_tool(
                action.get("action", ""), action.get("input", "")
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})

        return {
            "result": "Max steps reached. Partial results in steps.",
            "steps": steps if verbose else [],
            "agents_used": agents or ["default"],
            "tools_used": [],
        }

    def _exec_tool(self, tool: str, inp: str) -> str:
        try:
            if tool == "run_command":
                r = subprocess.run(
                    inp,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd="/home/lorai",
                )
                return (r.stdout + r.stderr)[:3000]
            elif tool == "read_file":
                return Path(os.path.expanduser(inp.strip())).read_text(
                    errors="ignore"
                )[:3000]
            elif tool == "write_file":
                lines = inp.split("\n", 1)
                if len(lines) == 2:
                    Path(lines[0].strip()).write_text(lines[1])
                    return f"Written to {lines[0].strip()}"
                return "Error: line 1 = path, rest = content"
            elif tool == "search_files":
                r = subprocess.run(
                    f'find /home/lorai -iname "*{inp}*" 2>/dev/null | head -20',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return r.stdout or "No files found"
            elif tool == "knowledge_search":
                from src.knowledge.knowledge_service import KnowledgeService

                results = KnowledgeService().search(inp, top_k=3, threshold=0.5)
                return (
                    json.dumps(results, indent=2)[:3000] if results else "No results"
                )
            else:
                return f"Unknown tool: {tool}"
        except Exception as e:
            return f"Error: {str(e)}"
