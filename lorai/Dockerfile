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
    && rm -rf /var/cache/apk/*

# noVNC + websockify
RUN apk add --no-cache novnc py3-websockify \
    || (pip3 install --break-system-packages websockify \
        && mkdir -p /usr/share/novnc \
        && curl -fsSL https://github.com/novnc/noVNC/archive/refs/tags/v1.4.0.tar.gz \
        | tar -xz -C /usr/share/novnc --strip-components=1)

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

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
