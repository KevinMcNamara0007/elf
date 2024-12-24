# Stage 1: Base Image Selection (Based on CUDA availability)
ARG IMAGE

# Check if the IMAGE argument is set correctly
FROM ${IMAGE} AS base


FROM --platform=linux/amd64 python:3.11-slim AS app

# Install git and cmake dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for the directory paths
ENV LLAMA_CPP_HOME=/opt/cx_intelligence/aiaas/compiled_llama_cpp
ENV LLAMA_SOURCE_FOLDER=/opt/cx_intelligence/aiaas/llama_source

# Create the directory for compiled binaries if it doesn't exist
RUN mkdir -p ${LLAMA_CPP_HOME}

# Set the working directory for the application
WORKDIR /app

# Copy the llama-server binary and shared libraries into the specific directory
COPY --from=base /app/llama-server ${LLAMA_CPP_HOME}

# Install runtime Python dependencies
COPY requirements.txt requirements.txt

# Check if the image contains GPU support and install the appropriate packages
RUN if echo "${IMAGE}" | grep -q "cuda"; then \
        pip install --no-cache-dir --upgrade pip setuptools wheel onnxruntime-gpu onnxruntime-genai-cuda; \
    else \
        pip install --no-cache-dir --upgrade pip setuptools wheel onnxruntime onnxruntime-genai; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container
COPY . .

# Expose the necessary ports
EXPOSE 8000-8010

# Set the entry point for the application
CMD ["uvicorn", "src.asgi:elf", "--host=0.0.0.0", "--port=8000"]
