# LorAI

**All of AI. One Command. Port 1842.**

LorAI Workspace is an AI-first operating environment running in Docker. It provides:

- OpenAI-compatible API gateway on port 1842
- Ollama-powered local LLMs
- Browser-accessible desktop via noVNC on port 6080
- AI Shell for natural language system commands
- Semantic file manager with ChromaDB

## Quick Start

```bash
docker run -d --name lorai \
  -p 1842:1842 \
  -p 6080:6080 \
  gajapathiks/lorai-workspace:latest
```

Then:
- API: http://localhost:1842
- Desktop: http://localhost:6080

## Or use the SDK

```bash
pip install lorai-workspace
```

```python
from lorai_workspace import LorAI
ai = LorAI()
print(ai.chat("Hello!"))
```

## License

MIT
