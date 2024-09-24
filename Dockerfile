# Base image with Ubuntu 22.04
FROM ubuntu:22.04 AS env-build

# Environment variables for paths
ENV LLAMA_CPP_HOME=/opt/cx_intelligence/aiaas/compiled_llama_cpp
ENV LLAMA_SOURCE_FOLDER=/opt/cx_intelligence/aiaas/llama_source

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
    libopenblas-dev \
    curl

# Clone the llama.cpp repository
RUN git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_SOURCE_FOLDER}

# Upgrade pip and install Python packages
RUN pip install --upgrade pip setuptools wheel

# Install CPU-only versions of TensorFlow and Keras
RUN pip install keras==2.15.0 tensorflow-cpu==2.15.0 pysqlite3-binary

# Set the working directory to the compiled llama_cpp folder
WORKDIR ${LLAMA_CPP_HOME}

# Build llama.cpp with OpenBLAS support (no GPU, CPU optimized)
RUN cmake -S ${LLAMA_SOURCE_FOLDER} -B . \
    -DGGML_OPENBLAS=ON \
    -DCMAKE_BUILD_TYPE=Release

RUN cmake --build . --config Release --target llama-server -j$(nproc)

# Stage for application
FROM env-build AS app

WORKDIR /app

COPY requirements.txt requirements.txt

# Install project requirements
RUN pip install -r requirements.txt

COPY . .

# Expose the necessary port (if applicable)
EXPOSE 8000-8010

# Set the entry point (if needed)
CMD ["uvicorn", "src.asgi:elf", "--host=0.0.0.0", "--port=8000"]
