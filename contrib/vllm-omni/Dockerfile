FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin
RUN apt-get update && \
    apt-get install -y git sox libsox-fmt-all jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/vllm-project/vllm-omni.git
RUN bash -c 'uv pip install -e ./vllm-omni --overrides <(echo "diffusers==0.35.0.dev0")'
