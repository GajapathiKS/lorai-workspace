"""LorAI CLI — All of AI. One Command. Port 1842.

Usage:
    lorai-workspace start [--gpu] [--port PORT]
    lorai-workspace stop
    lorai-workspace status
    lorai-workspace chat "message"
    lorai-workspace desktop
    lorai-workspace pull <model>
    lorai-workspace bench
    lorai-workspace logs
    lorai-workspace version
    lorai-workspace help
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import webbrowser

import httpx

BANNER = r"""
  ██╗      ██████╗  ██████╗  █████╗ ██╗
  ██║     ██╔═══██╗██╔══██╗██╔══██╗██║
  ██║     ██║   ██║██████╔╝███████║██║
  ██║     ██║   ██║██╔══██╗██╔══██║██║
  ███████╗╚██████╔╝██║  ██║██║  ██║██║
  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝

  All of AI. One Command. Port 1842.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lorai",
        description="LorAI — All of AI. One Command.",
    )
    sub = parser.add_subparsers(dest="command")

    # start
    p_start = sub.add_parser("start", help="Start the LorAI container")
    p_start.add_argument("--gpu", action="store_true", help="Enable GPU passthrough")
    p_start.add_argument("--port", type=int, default=1842, help="API port (default: 1842)")

    # stop
    sub.add_parser("stop", help="Stop the LorAI container")

    # status
    sub.add_parser("status", help="Show LorAI status")

    # chat
    p_chat = sub.add_parser("chat", help="Chat with the local AI")
    p_chat.add_argument("message", help="Message to send")
    p_chat.add_argument("--model", default=None, help="Model to use")

    # desktop
    sub.add_parser("desktop", help="Open LorAI desktop in browser")

    # pull
    p_pull = sub.add_parser("pull", help="Pull/download an AI model")
    p_pull.add_argument("model", help="Model name (e.g. llama3, phi3:mini)")

    # bench
    sub.add_parser("bench", help="Benchmark your hardware")

    # logs
    sub.add_parser("logs", help="Show container logs")

    # version
    sub.add_parser("version", help="Show LorAI version")

    # help (explicit)
    sub.add_parser("help", help="Show help with banner")

    args = parser.parse_args(argv)

    if args.command is None or args.command == "help":
        _cmd_help()
    elif args.command == "start":
        _cmd_start(port=args.port, gpu=args.gpu)
    elif args.command == "stop":
        _cmd_stop()
    elif args.command == "status":
        _cmd_status()
    elif args.command == "chat":
        _cmd_chat(args.message, model=args.model)
    elif args.command == "desktop":
        _cmd_desktop()
    elif args.command == "pull":
        _cmd_pull(args.model)
    elif args.command == "bench":
        _cmd_bench()
    elif args.command == "logs":
        _cmd_logs()
    elif args.command == "version":
        _cmd_version()


def _cmd_help() -> None:
    print(BANNER)
    print("Usage: lorai <command> [options]\n")
    print("Commands:")
    print("  start [--gpu] [--port N]   Start the LorAI container")
    print("  stop                       Stop the LorAI container")
    print("  status                     Show system status")
    print('  chat "message"             Chat with the local AI')
    print("  desktop                    Open browser desktop (noVNC)")
    print("  pull <model>               Download an AI model")
    print("  bench                      Benchmark your hardware")
    print("  logs                       Show container logs")
    print("  version                    Show version info")
    print("  help                       Show this help")
    print()
    print("Examples:")
    print("  lorai-workspace start")
    print('  lorai-workspace chat "What is the meaning of life?"')
    print("  lorai-workspace pull llama3")
    print("  lorai-workspace desktop")
    print("  lorai-workspace stop")


def _cmd_start(port: int = 1842, gpu: bool = False) -> None:
    from lorai.docker import ensure_running
    ensure_running(port=port, gpu=gpu)
    print(f"\nLorAI is running at http://localhost:{port}")
    print("Desktop available at http://localhost:6080")


def _cmd_stop() -> None:
    from lorai.docker import stop_container
    stop_container()
    print("LorAI stopped.")


def _cmd_status() -> None:
    from lorai.docker import status
    s = status()
    print(BANNER)
    print(f"  Docker installed:    {'yes' if s['docker_installed'] else 'NO'}")
    print(f"  Image pulled:        {'yes' if s['image_pulled'] else 'no'}")
    print(f"  Container running:   {'yes' if s['container_running'] else 'no'}")
    print(f"  API healthy:         {'yes' if s['api_healthy'] else 'no'}")
    print(f"  API URL:             {s['url']}")
    print(f"  Desktop URL:         {s['desktop_url']}")


def _cmd_chat(message: str, model: str | None = None) -> None:
    payload = {
        "model": model or "auto",
        "messages": [{"role": "user", "content": message}],
    }
    try:
        resp = httpx.post(
            "http://localhost:1842/v1/chat/completions",
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        print(data["choices"][0]["message"]["content"])
    except httpx.ConnectError:
        print("Error: LorAI is not running. Start it with: lorai-workspace start")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def _cmd_desktop() -> None:
    url = "http://localhost:6080"
    print(f"Opening LorAI desktop: {url}")
    webbrowser.open(url)


def _cmd_pull(model: str) -> None:
    print(f"Pulling model: {model}")
    try:
        resp = httpx.post(
            "http://localhost:1842/lorai/hub/pull",
            json={"name": model}, timeout=600,
        )
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except httpx.ConnectError:
        print("Error: LorAI is not running. Start it with: lorai-workspace start")
        sys.exit(1)


def _cmd_bench() -> None:
    print("Running benchmark...")
    try:
        resp = httpx.post(
            "http://localhost:1842/lorai/hub/bench",
            json={}, timeout=300,
        )
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except httpx.ConnectError:
        print("Error: LorAI is not running. Start it with: lorai-workspace start")
        sys.exit(1)


def _cmd_logs() -> None:
    try:
        subprocess.run(
            ["docker", "logs", "--tail", "50", "-f", "lorai"],
            check=True,
        )
    except FileNotFoundError:
        print("Error: Docker is not installed.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: LorAI container not found. Start it with: lorai-workspace start")
        sys.exit(1)


def _cmd_version() -> None:
    from lorai import __version__, PORT
    print(BANNER)
    print(f"  Version: {__version__}")
    print(f"  Port:    {PORT}")
    print(f"  URL:     http://localhost:{PORT}")
    print("  Desktop: http://localhost:6080")
