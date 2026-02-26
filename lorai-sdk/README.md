# LorAI SDK

**All of AI. One Command. Port 1842.**

LorAI gives you a local, free, OpenAI-compatible AI platform with 50+ tools — LLMs, image gen, video, voice, code, agents, RAG, vision — all running in a Docker container on port 1842.

## Quick Start

```bash
pip install lorai-workspace
```

```python
from lorai import LorAI

ai = LorAI()              # auto-pulls Docker image, starts container
print(ai.chat("Hello!"))   # gets response from local LLM
```

## CLI

```bash
lorai-workspace start          # Start the LorAI container
lorai-workspace chat "Hello!"  # Chat with your local AI
lorai-workspace status         # Check system status
lorai-workspace desktop        # Open the browser desktop
lorai-workspace stop           # Stop the container
```

## Features

- **OpenAI-compatible**: Drop-in replacement for `openai.OpenAI()`
- **Auto-managed Docker**: Container pulls and starts automatically
- **10 AI services**: Chat, Image, Video, Voice, Knowledge, Agents, Code, Vision, LoRA, Hub
- **Port 1842**: Named after Ada Lovelace's year

## License

MIT
