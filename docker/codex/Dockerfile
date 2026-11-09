FROM node:22-bookworm-slim
ARG CODEX_VERSION=0.158.0
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g --include=optional "@openai/codex@${CODEX_VERSION}" \
    && codex exec --help >/dev/null \
    && usermod --login agent --home /home/agent --move-home node
COPY docker/codex/entrypoint.py /opt/agent/entrypoint.py
COPY graynet/tools/stdio.py /opt/agent/mcp_stdio.py
COPY graynet/runtime/egress_proxy.py /opt/agent/egress_proxy.py
USER 1000:1000
WORKDIR /scratch
ENV CODEX_HOME=/home/agent/.codex
ENTRYPOINT ["python3", "-I", "/opt/agent/entrypoint.py"]
