FROM alpine:3.19

# System packages
RUN apk add --no-cache \
    bash \
    curl \
    git \
    sudo \
    supervisor \
    xvfb \
    x11vnc \
    openbox \
    font-noto \
    font-noto-emoji \
    ttf-dejavu \
    pcmanfm \
    xterm \
    python3 \
    py3-pip \
    ffmpeg \
    nodejs \
    make \
    cmake \
    g++ \
    zstd \
    && rm -rf /var/cache/apk/*

# noVNC + websockify
RUN apk add --no-cache novnc py3-websockify \
    || (pip3 install --break-system-packages websockify \
        && mkdir -p /usr/share/novnc \
        && curl -fsSL https://github.com/novnc/noVNC/archive/refs/tags/v1.4.0.tar.gz \
        | tar -xz -C /usr/share/novnc --strip-components=1)

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Whisper.cpp — Speech to Text (soft dependency — build can fail on minimal Alpine)
RUN git clone https://github.com/ggerganov/whisper.cpp /opt/whisper.cpp \
    && cd /opt/whisper.cpp \
    && cmake -B build && cmake --build build -j$(nproc) \
    && bash models/download-ggml-model.sh base.en \
    || echo "Whisper.cpp skipped (install later with: lorai install whisper)"

# Piper TTS — Text to Speech (soft dependency — Alpine onnxruntime can be tricky)
RUN pip3 install --break-system-packages onnxruntime piper-tts \
    && mkdir -p /data/voices && cd /data/voices \
    && curl -L -o en_US-amy-medium.onnx \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx" \
    && curl -L -o en_US-amy-medium.onnx.json \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json" \
    || echo "Piper TTS skipped (install later with: pip install piper-tts)"

# Python dependencies (core)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Image generation dependencies (loaded lazily on first GPU request)
# Installed separately so CPU-only builds can skip/ignore failures
RUN pip3 install --break-system-packages --no-cache-dir \
    diffusers torch torchvision accelerate safetensors \
    || echo "GPU dependencies skipped (CPU-only build)"

# Create lorai user
RUN adduser -D -s /bin/bash lorai \
    && echo "lorai ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create directories
RUN mkdir -p \
    /home/lorai/Desktop \
    /home/lorai/Documents \
    /home/lorai/Downloads \
    /data/vectors \
    /data/models \
    /data/config \
    /data/loras \
    && chown -R lorai:lorai /home/lorai /data

# Copy application files
COPY config/ /opt/lorai/config/
COPY src/ /opt/lorai/src/
COPY scripts/ /opt/lorai/scripts/
RUN chmod +x /opt/lorai/scripts/entrypoint.sh \
    && chown -R lorai:lorai /opt/lorai

# Environment
ENV DISPLAY=:1 \
    VNC_PORT=5900 \
    NOVNC_PORT=6080 \
    LORAI_PORT=1842 \
    LORAI_MODE=hybrid \
    LORAI_MODEL=phi3:mini \
    HOME=/home/lorai

EXPOSE 1842 6080

USER lorai
WORKDIR /home/lorai

ENTRYPOINT ["/opt/lorai/scripts/entrypoint.sh"]
