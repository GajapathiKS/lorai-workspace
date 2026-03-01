"""Docker container management for LorAI.

Handles pulling, starting, stopping, and health-checking the
LorAI Desktop Docker container.
"""

import os
import shutil
import subprocess
import sys
import time

import httpx

IMAGE = os.environ.get("LORAI_IMAGE", "gajapathiks/lorai-workspace:latest")
CONTAINER_NAME = "lorai"
HEALTH_TIMEOUT = 120  # seconds to wait for healthy state


def is_docker_installed() -> bool:
    """Check if the docker CLI is available on PATH."""
    return shutil.which("docker") is not None


def is_daemon_running() -> bool:
    """Check if the Docker daemon is running and accepting connections."""
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def is_image_pulled() -> bool:
    """Check if the LorAI Desktop image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", IMAGE],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def is_container_running() -> bool:
    """Check if a container named 'lorai' is currently running."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True, text=True,
    )
    return result.stdout.strip() == "true"


def is_lorai_healthy(port: int = 1842) -> bool:
    """Check if the LorAI API responds on the given port."""
    try:
        resp = httpx.get(f"http://localhost:{port}/api/health", timeout=5)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def pull_image() -> None:
    """Pull the LorAI Desktop Docker image."""
    print("Pulling LorAI Docker image (this may take a while)...")
    subprocess.run(["docker", "pull", IMAGE], check=True)


def start_container(port: int = 1842, vnc_port: int = 6080, gpu: bool = False) -> None:
    """Start the LorAI Docker container."""
    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "-p", f"{port}:1842",
        "-p", f"{vnc_port}:6080",
        "-v", f"{_data_dir()}:/data",
        "-e", "LORAI_MODE=hybrid",
        "-e", "LORAI_MODEL=phi3:mini",
    ]
    if gpu:
        cmd.extend(["--gpus", "all"])
    cmd.append(IMAGE)

    print(f"Starting LorAI container on port {port}...")
    subprocess.run(cmd, check=True)


def stop_container() -> None:
    """Stop and remove the LorAI container."""
    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)


def ensure_running(port: int = 1842, gpu: bool = False) -> None:
    """Orchestrate: check -> pull -> start -> health check."""
    if not is_docker_installed():
        raise RuntimeError(
            "Docker is not installed.\n"
            "Download and install Docker Desktop: https://docs.docker.com/get-docker/"
        )

    if not is_daemon_running():
        if sys.platform == "win32":
            raise RuntimeError(
                "Docker Desktop is not running.\n"
                "Open Docker Desktop from the Start menu, wait for it to finish starting "
                "(whale icon in taskbar turns solid), then run your script again."
            )
        elif sys.platform == "darwin":
            raise RuntimeError(
                "Docker Desktop is not running.\n"
                "Open Docker Desktop from Applications, wait for it to finish starting "
                "(whale icon in menu bar turns solid), then run your script again."
            )
        else:
            raise RuntimeError(
                "Docker daemon is not running.\n"
                "Start it with: sudo systemctl start docker"
            )

    if is_container_running() and is_lorai_healthy(port):
        return

    # Stop stale container if exists
    if is_container_running():
        stop_container()

    if not is_image_pulled():
        pull_image()

    start_container(port=port, gpu=gpu)

    # Wait for health
    print("Waiting for LorAI to become ready...")
    deadline = time.time() + HEALTH_TIMEOUT
    while time.time() < deadline:
        if is_lorai_healthy(port):
            print(f"LorAI Workspace is ready at http://localhost:{port}")
            return
        time.sleep(2)

    raise RuntimeError(
        f"LorAI did not become healthy within {HEALTH_TIMEOUT}s. "
        "Check logs with: docker logs lorai"
    )


def status(port: int = 1842) -> dict:
    """Return a dict describing the current state of LorAI."""
    return {
        "docker_installed": is_docker_installed(),
        "image_pulled": is_image_pulled() if is_docker_installed() else False,
        "container_running": is_container_running() if is_docker_installed() else False,
        "api_healthy": is_lorai_healthy(port),
        "port": port,
        "url": f"http://localhost:{port}",
        "desktop_url": "http://localhost:6080",
    }


def _data_dir() -> str:
    """Return the host-side data directory (~/.lorai/data)."""
    from pathlib import Path
    data = Path.home() / ".lorai" / "data"
    data.mkdir(parents=True, exist_ok=True)
    return str(data)
