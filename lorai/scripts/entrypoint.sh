#!/bin/bash
set -e

BANNER='
  ██╗      ██████╗  ██████╗  █████╗ ██╗
  ██║     ██╔═══██╗██╔══██╗██╔══██╗██║
  ██║     ██║   ██║██████╔╝███████║██║
  ██║     ██║   ██║██╔══██╗██╔══██║██║
  ███████╗╚██████╔╝██║  ██║██║  ██║██║
  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝

  All of AI. One Command. Port 1842.
'

echo "$BANNER"
echo "  Starting LorAI..."
echo ""

# Add src to PYTHONPATH for service imports
export PYTHONPATH="/opt/lorai:${PYTHONPATH}"

# Ensure data directories exist
mkdir -p /data/vectors /data/models /data/config /data/loras /data/voices

# 1. Start Xvfb (virtual display)
echo "[1/8] Starting virtual display..."
Xvfb :1 -screen 0 1280x720x24 &
sleep 1

# 2. Start Openbox (window manager)
echo "[2/8] Starting window manager..."
openbox --config-file /opt/lorai/config/openbox-rc.xml &
sleep 1

# 3. Start x11vnc (VNC server)
echo "[3/8] Starting VNC server..."
x11vnc -display :1 -nopw -listen 0.0.0.0 -rfbport 5900 -shared -forever -bg -o /tmp/x11vnc.log 2>/dev/null

# 4. Start websockify/noVNC (browser access on port 6080)
echo "[4/8] Starting noVNC (browser desktop)..."
NOVNC_DIR=""
for d in /usr/share/novnc /usr/share/webapps/novnc /usr/share/noVNC; do
    if [ -d "$d" ]; then
        NOVNC_DIR="$d"
        break
    fi
done

if [ -n "$NOVNC_DIR" ]; then
    websockify --web="$NOVNC_DIR" 6080 localhost:5900 &
else
    echo "  Warning: noVNC not found. Browser desktop unavailable."
    websockify 6080 localhost:5900 &
fi

# 5. Start Ollama server
echo "[5/8] Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "  Waiting for Ollama..."
for i in $(seq 1 60); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  Ollama is ready."
        break
    fi
    sleep 2
done

# 6. Pull default model (only if not already pulled)
echo "[6/8] Checking default model (${LORAI_MODEL})..."
MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null || echo '{"models":[]}')
if echo "$MODELS" | grep -q "${LORAI_MODEL}"; then
    echo "  Model ${LORAI_MODEL} already available."
else
    echo "  Pulling ${LORAI_MODEL} (this may take a while)..."
    ollama pull "${LORAI_MODEL}" || echo "  Warning: Failed to pull model. You can pull it later with: lorai pull ${LORAI_MODEL}"
fi

# 7. Start LorAI Gateway
echo "[7/8] Starting LorAI API Gateway on port ${LORAI_PORT}..."
python3 /opt/lorai/src/gateway/gateway.py &

# 8. Start LorAI Shell and Desktop components
echo "[8/8] Starting LorAI Shell and Desktop..."
python3 /opt/lorai/src/ai-shell/lorai_shell.py &
python3 /opt/lorai/src/desktop/lorai_panel.py &

# Launch terminal on desktop
sleep 2
xterm -fa 'Monospace' -fs 12 -title "LorAI Terminal" \
    -e "python3 /opt/lorai/src/ai-shell/lorai_terminal.py" &

# Start file manager daemon
python3 /opt/lorai/src/file-manager/lorai_fs.py &

# Detect hardware profile
HW_PROFILE="cpu"
if command -v nvidia-smi &> /dev/null; then
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$VRAM" ]; then
        VRAM_GB=$((VRAM / 1024))
        if [ "$VRAM_GB" -ge 24 ]; then
            HW_PROFILE="power"
        elif [ "$VRAM_GB" -ge 8 ]; then
            HW_PROFILE="standard"
        else
            HW_PROFILE="lite"
        fi
    fi
fi

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║           LorAI is Ready!                ║"
echo "  ║                                          ║"
echo "  ║  API:     http://localhost:${LORAI_PORT}         ║"
echo "  ║  Desktop: http://localhost:6080          ║"
echo "  ║  Health:  http://localhost:${LORAI_PORT}/api/health ║"
echo "  ║  Profile: ${HW_PROFILE}                           ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  Services: chat, code, voice, knowledge, agents, vision"
echo "  GPU services: image, video, music (requires --gpu)"
echo ""

# Keep container alive
wait
