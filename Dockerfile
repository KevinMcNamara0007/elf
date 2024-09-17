# Use the NVIDIA CUDA base image with CUDA 12.6
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS base

# Install necessary packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    software-properties-common \
    gnupg \
    python3 \
    python3-pip \
    qemu-user-static

# Download and install cuDNN
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.4.0/local_installers/cudnn-local-repo-ubuntu2204-9.4.0_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2204-9.4.0_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-9.4.0/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y cudnn-cuda-12 && \
    rm cudnn-local-repo-ubuntu2204-9.4.0_1.0-1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the EFS folder where the models are stored
COPY efs /efs

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    rm -rf /root/.cache/pip

# Install onnxruntime-genai based on GPU availability
RUN if nvcc --version > /dev/null 2>&1; then \
        echo "GPU detected, installing onnxruntime-genai-cuda and onnxruntime-gpu" && \
        pip install onnxruntime-genai-cuda==0.4.0 onnxruntime-gpu==1.19.2; \
    else \
        echo "No GPU detected, installing CPU version of onnxruntime-genai" && \
        pip install onnxruntime-genai onnxruntime; \
    fi

# Set the working directory
WORKDIR /app

# Copy the rest of the application code
COPY . .

# Expose necessary ports
EXPOSE 8000

# Set the entry point to the script
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["uvicorn", "src.asgi:elf", "--host=0.0.0.0", "--port=8000"]
