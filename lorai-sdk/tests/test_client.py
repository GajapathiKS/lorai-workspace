"""Tests for the LorAI SDK."""

from unittest.mock import patch


# ------------------------------------------------------------------
# Test constants
# ------------------------------------------------------------------

def test_port_is_1842():
    import lorai_workspace
    assert lorai_workspace.PORT == 1842


def test_version_is_set():
    import lorai_workspace
    assert lorai_workspace.__version__ == "0.1.1"


# ------------------------------------------------------------------
# Test LorAI client
# ------------------------------------------------------------------

@patch("lorai_workspace.client.ensure_running")
def test_lorai_inherits_from_openai(mock_ensure):
    from openai import OpenAI
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    assert isinstance(ai, OpenAI)


@patch("lorai_workspace.client.ensure_running")
def test_auto_start_calls_ensure_running(mock_ensure):
    from lorai_workspace import LorAI
    LorAI(auto_start=True)
    mock_ensure.assert_called_once_with(port=1842, gpu=False)


@patch("lorai_workspace.client.ensure_running")
def test_auto_start_false_skips_ensure_running(mock_ensure):
    from lorai_workspace import LorAI
    LorAI(auto_start=False)
    mock_ensure.assert_not_called()


@patch("lorai_workspace.client.ensure_running")
def test_custom_port_passes_through(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=True, port=9999)
    mock_ensure.assert_called_once_with(port=9999, gpu=False)
    assert ai._port == 9999


@patch("lorai_workspace.client.ensure_running")
def test_gpu_flag_passes_through(mock_ensure):
    from lorai_workspace import LorAI
    LorAI(auto_start=True, gpu=True)
    mock_ensure.assert_called_once_with(port=1842, gpu=True)


# ------------------------------------------------------------------
# Test all 9 services exist with correct methods
# ------------------------------------------------------------------

@patch("lorai_workspace.client.ensure_running")
def test_image_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.image
    assert hasattr(svc, "generate")
    assert hasattr(svc, "edit")


@patch("lorai_workspace.client.ensure_running")
def test_video_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.video
    assert hasattr(svc, "generate")


@patch("lorai_workspace.client.ensure_running")
def test_voice_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.voice
    assert hasattr(svc, "transcribe")
    assert hasattr(svc, "speak")


@patch("lorai_workspace.client.ensure_running")
def test_knowledge_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.knowledge
    assert hasattr(svc, "ingest")
    assert hasattr(svc, "search")
    assert hasattr(svc, "ask")


@patch("lorai_workspace.client.ensure_running")
def test_agents_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.agents
    assert hasattr(svc, "run")
    assert hasattr(svc, "list_agents")
    assert hasattr(svc, "list_tools")


@patch("lorai_workspace.client.ensure_running")
def test_code_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.code
    assert hasattr(svc, "generate")
    assert hasattr(svc, "review")


@patch("lorai_workspace.client.ensure_running")
def test_vision_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.vision
    assert hasattr(svc, "analyze")
    assert hasattr(svc, "ocr")


@patch("lorai_workspace.client.ensure_running")
def test_lora_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.lora
    assert hasattr(svc, "list")
    assert hasattr(svc, "load")
    assert hasattr(svc, "unload")


@patch("lorai_workspace.client.ensure_running")
def test_hub_service(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    svc = ai.hub
    assert hasattr(svc, "models")
    assert hasattr(svc, "pull")
    assert hasattr(svc, "remove")
    assert hasattr(svc, "status")
    assert hasattr(svc, "bench")


# ------------------------------------------------------------------
# Test docker.py functions exist
# ------------------------------------------------------------------

def test_docker_module_functions():
    from lorai_workspace import docker
    assert callable(docker.is_docker_installed)
    assert callable(docker.is_image_pulled)
    assert callable(docker.is_container_running)
    assert callable(docker.is_lorai_healthy)
    assert callable(docker.pull_image)
    assert callable(docker.start_container)
    assert callable(docker.stop_container)
    assert callable(docker.ensure_running)
    assert callable(docker.status)


# ------------------------------------------------------------------
# Test CLI entry point
# ------------------------------------------------------------------

def test_cli_entry_point():
    """Verify the CLI main function is importable and callable."""
    from lorai_workspace.cli import main
    assert callable(main)


def test_cli_version_command(capsys):
    """Test that 'lorai-workspace version' prints version and port."""
    from lorai_workspace.cli import main
    main(["version"])
    captured = capsys.readouterr()
    import lorai_workspace
    assert lorai_workspace.__version__ in captured.out
    assert "1842" in captured.out


def test_cli_help_command(capsys):
    """Test that 'lorai-workspace help' prints the banner."""
    from lorai_workspace.cli import main
    main(["help"])
    captured = capsys.readouterr()
    assert "All of AI" in captured.out
    assert "1842" in captured.out


# ------------------------------------------------------------------
# Test default model
# ------------------------------------------------------------------

@patch("lorai_workspace.client.ensure_running")
def test_default_model(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False, default_model="llama3")
    assert ai._default_model == "llama3"


@patch("lorai_workspace.client.ensure_running")
def test_default_model_auto(mock_ensure):
    from lorai_workspace import LorAI
    ai = LorAI(auto_start=False)
    assert ai._default_model == "auto"
