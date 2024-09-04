FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 AS env-build

ENV LLAMA_CPP_HOME=/opt/cx_intelligence/aiaas/compiled_llama_cpp
ENV LLAMA_SOURCE_FOLDER=/opt/cx_intelligence/aiaas/llama_source

RUN mkdir -p ${LLAMA_SOURCE_FOLDER} \
    && mkdir -p ${LLAMA_CPP_HOME}

# Install build tools, CUDA libraries, and clone llama.cpp
RUN apt-get update && apt-get install -y build-essential git libgomp1 cmake

# Clone the llama.cpp repository
RUN git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_SOURCE_FOLDER}

# Set the working directory to the compiled llama_cpp folder
WORKDIR ${LLAMA_CPP_HOME}

# Set CUDA library paths for linking
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# Build llama.cpp with CUDA support
RUN cmake -S ${LLAMA_SOURCE_FOLDER} -B . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86"

RUN cmake --build . --config Release --target llama-server
