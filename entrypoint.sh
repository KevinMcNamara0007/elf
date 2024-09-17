#!/bin/bash

# Paths to the CPU and CUDA model directories
CPU_MODEL_DIR="/efs/models/microsoft/Phi-3-vision-128k-instruct-onnx-cpu"
CUDA_MODEL_DIR="/efs/models/microsoft/Phi-3-vision-128k-instruct-onnx-cuda"
TARGET_MODEL_DIR="/app/model"

# Check if GPU is available
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "GPU detected, using CUDA model"
    cp -r "$CUDA_MODEL_DIR"/* "$TARGET_MODEL_DIR/"
    echo "Cleaning up CPU model to save space"
    rm -rf "$CPU_MODEL_DIR"
else
    echo "No GPU detected, using CPU model"
    cp -r "$CPU_MODEL_DIR"/* "$TARGET_MODEL_DIR/"
    echo "Cleaning up CUDA model to save space"
    rm -rf "$CUDA_MODEL_DIR"
fi

# Proceed with starting the application
exec "$@"
