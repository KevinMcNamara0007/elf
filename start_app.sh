#!/bin/bash
# Start the llama.cpp server
python -c "from src.utilities.general import start_llama_cpp; start_llama_cpp()"

# Start the Uvicorn server
uvicorn src.asgi:elf --host $HOST --port $UVICORN_PORT
