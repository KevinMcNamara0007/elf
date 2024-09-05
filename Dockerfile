# Use the NVIDIA CUDA base image with development tools installed
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS env-build

# Environment variables for paths
ENV LLAMA_CPP_HOME=/opt/cx_intelligence/aiaas/compiled_llama_cpp
ENV LLAMA_SOURCE_FOLDER=/opt/cx_intelligence/aiaas/llama_source
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# Create necessary directories
RUN mkdir -p ${LLAMA_SOURCE_FOLDER} && mkdir -p ${LLAMA_CPP_HOME}

# Install build tools and dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    cmake \
    python3 \
    python3-pip \
    nvidia-container-toolkit

# Clone the llama.cpp repository
RUN git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_SOURCE_FOLDER}

# Upgrade pip and install Python packages
RUN pip install --upgrade pip setuptools wheel

RUN pip install keras==2.15.0 tensorflow==2.15.0

# Set the working directory to the compiled llama_cpp folder
WORKDIR ${LLAMA_CPP_HOME}

# Verify CUDA library path
RUN ls /usr/local/cuda/lib64/stubs

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

# Build llama.cpp with CUDA support and specify CUDA architectures
RUN cmake -S ${LLAMA_SOURCE_FOLDER} -B . \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-L/usr/local/cuda/lib64 -lcuda" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64 -lcuda" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -DCMAKE_INSTALL_RPATH=/usr/local/cuda/lib64

RUN cmake --build . --config Release --target llama-server -j$(nproc)

RUN nvidia-ctk runtime configure --runtime=docker && systemctl restart docker

FROM env-build AS app

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

# Expose the necessary port (if applicable)
EXPOSE 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010

# Set the entry point (if needed)
CMD ["uvicorn", "src.asgi:elf", "--host=0.0.0.0", "--port=8000"]
