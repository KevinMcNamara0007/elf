#!/bin/bash

# Check if CUDA is available by checking for nvcc (the CUDA compiler)
if command -v nvcc &> /dev/null; then
  IMAGE="ghcr.io/ggerganov/llama.cpp:server-cuda"
  echo "CUDA is available. Using server-cuda image... ${IMAGE}"
else
  IMAGE="ghcr.io/ggerganov/llama.cpp:server"
  echo "No CUDA found. Using server image... ${IMAGE}"
fi

# Build the Docker image with the selected base image
docker build --platform linux/amd64 --build-arg IMAGE=${IMAGE} -t my-llama-app .
