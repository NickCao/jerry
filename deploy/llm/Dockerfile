FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin

# Download model weights at build time for offline use
RUN huggingface-cli download Qwen/Qwen2.5-3B-Instruct-AWQ
