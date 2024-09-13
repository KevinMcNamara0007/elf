# Use the NVIDIA CUDA base image with CUDA 12.6
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

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
    python3-pip

# Download and install the cuDNN 9.4.0 package
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.4.0/local_installers/cudnn-local-repo-ubuntu2204-9.4.0_1.0-1_amd64.deb \
    && dpkg -i cudnn-local-repo-ubuntu2204-9.4.0_1.0-1_amd64.deb

# Copy the keyring to the correct location
RUN cp /var/cudnn-local-repo-ubuntu2204-9.4.0/cudnn-*-keyring.gpg /usr/share/keyrings/

# Update and install cuDNN for CUDA 12
RUN apt-get update && \
    apt-get install -y cudnn-cuda-12

# Set the library path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify installation
RUN echo "Checking CUDA version" && nvcc --version
RUN echo "Checking cuDNN installation" && dpkg -l | grep cudnn

# Set the working directory
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

# Expose the necessary port (if applicable)
EXPOSE 8000-8010

# Set the entry point (if needed)
CMD ["uvicorn", "src.asgi:elf", "--host=0.0.0.0", "--port=8000"]
