FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_TOOL_BIN_DIR=/usr/local/bin

RUN apt-get update && \
    apt-get install -y libxcb1 libgl1 libglib2.0-0t64 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"

# Pre-download models at build time for offline use
RUN python3 -c "import whisper; whisper.load_model('tiny')"
RUN python3 -c "from pipecat.audio.vad.silero import SileroVADAnalyzer; SileroVADAnalyzer()"

ENTRYPOINT []
CMD ["uv", "run", "main.py"]
