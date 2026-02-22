"""LorAI Quickstart — All of AI. One Command."""

from lorai import LorAI

# Create client (auto-starts Docker container on first use)
ai = LorAI()

# Simple chat
response = ai.chat("What is LorAI?")
print(response)

# OpenAI-compatible usage (works with any OpenAI code)
completion = ai.chat.completions.create(
    model="phi3:mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(completion.choices[0].message.content)
