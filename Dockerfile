# Use the NVIDIA CUDA base image with Python 3.9
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS base_dep

# Set the working directory to /app
WORKDIR /app

# Environment variable to suppress tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Dallas

# Install necessary build tools, Git, CMake, and tzdata
RUN apt-get update && \
    apt-get install -y \
    git \
    cmake \
    build-essential \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9
RUN apt-get update && apt-get install -y python3.9 python3.9-venv python3.9-dev python3-pip && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

# Copy the contents of the current directory to /app in the container
COPY . .

# List the contents of /app to verify the files have been copied
RUN echo "Listing contents of /app after COPY in base_dep stage:" && ls -l

# Install TensorFlow with GPU support
RUN python3 -m pip install tensorflow==2.13.1 keras==2.13.1

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional Python packages from requirements file
RUN python3 -m pip install -r docker_requirements.txt

# Create a new shell script to start the application
RUN echo '#!/bin/bash\n\
# Start the llama.cpp server\n\
python -c "from src.utilities.general import start_llama_cpp; start_llama_cpp()"\n\
\n\
# Start the Uvicorn server\n\
uvicorn src.asgi:elf --host $HOST --port $UVICORN_PORT' > /app/start_app.sh

# Make the shell script executable
RUN chmod +x /app/start_app.sh

# Expose necessary ports
EXPOSE 8000 8001 8002 8003 8004

# Define environment variables
ENV HOST=0.0.0.0
ENV UVICORN_PORT=8000

# Run the shell script
CMD ["./start_app.sh"]
