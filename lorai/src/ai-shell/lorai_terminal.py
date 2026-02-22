"""LorAI Terminal — Rich-based interactive CLI.

Connects to the LorAI Shell backend via HTTP and provides
an interactive terminal for natural language commands.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import httpx

SHELL_URL = "http://127.0.0.1:8421"

BANNER = r"""
  ██╗      ██████╗  ██████╗  █████╗ ██╗
  ██║     ██╔═══██╗██╔══██╗██╔══██╗██║
  ██║     ██║   ██║██████╔╝███████║██║
  ██║     ██║   ██║██╔══██╗██╔══██║██║
  ███████╗╚██████╔╝██║  ██║██║  ██║██║
  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝

  All of AI. One Command. Port 1842.
  Type your request in natural language. Type 'exit' to quit.
"""


def main():
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    print(BANNER)
    context = []

    while True:
        try:
            user_input = input("\n🤖 lorai> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # Direct shell command passthrough
        if user_input.startswith("!"):
            cmd = user_input[1:].strip()
            if cmd:
                os.system(cmd)
            continue

        # Ask the shell backend
        try:
            resp = httpx.post(
                f"{SHELL_URL}/shell/ask",
                json={"message": user_input, "context": context[-10:]},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError:
            print("  [Shell backend not available — running in direct mode]")
            # Fallback: try to interpret as a direct command
            _fallback_chat(user_input)
            continue
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Display thought
        thought = data.get("thought", "")
        if thought:
            print(f"\n  💭 {thought}")

        # Execute actions
        for action in data.get("actions", []):
            action_type = action.get("type", "")
            value = action.get("value", "")

            if action_type == "say":
                if use_rich:
                    console.print(Markdown(value))
                else:
                    print(f"\n  {value}")
            elif action_type == "command":
                print(f"\n  $ {value}")
                confirm = input("  Run this command? [Y/n] ").strip().lower()
                if confirm in ("", "y", "yes"):
                    result = subprocess.run(value, shell=True, capture_output=True, text=True)
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(result.stderr)
            elif action_type == "browse":
                print(f"\n  Opening: {value}")
                os.system(f"xdg-open '{value}' 2>/dev/null &")

        # Add to context
        context.append({"role": "user", "content": user_input})
        context.append({"role": "assistant", "content": data.get("raw", "")})


def _fallback_chat(message: str):
    """Fallback: chat directly with Ollama when shell backend is unavailable."""
    try:
        resp = httpx.post(
            "http://localhost:1842/v1/chat/completions",
            json={
                "model": os.environ.get("LORAI_MODEL", "phi3:mini"),
                "messages": [{"role": "user", "content": message}],
            },
            timeout=120,
        )
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"\n  {content}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
