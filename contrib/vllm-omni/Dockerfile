FROM ghcr.io/nvidia-ai-iot/vllm:0.19.0-r36.4.tegra-aarch64-cu126-22.04
RUN apt-get update && \
    apt-get install -y git sox libsox-fmt-all jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/vllm-project/vllm-omni.git
RUN bash -c 'uv pip install -e ./vllm-omni --overrides <(echo "diffusers==0.35.0.dev0")'
