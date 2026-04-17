FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
RUN apt-get update && \
    apt-get install -y git sox libsox-fmt-all jq espeak-ng && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git clone --branch kokoro https://github.com/NickCao/vllm-omni.git
RUN bash -c 'uv pip install -e ./vllm-omni --overrides <(echo "diffusers==0.35.0.dev0")'

# Download model weights at build time for offline use
RUN huggingface-cli download hexgrad/Kokoro-82M
