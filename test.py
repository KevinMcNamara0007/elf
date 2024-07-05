import multiprocessing
import os
import platform
import subprocess
from contextlib import asynccontextmanager
import shutil

import signal
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
import requests

PLATFORM = platform.system()
CONTEXT_WINDOW = "32000"
HOST = "127.0.0.1"
UVICORN_PORT = 8000
LLAMA_PORT = UVICORN_PORT + 1
LLAMA_CPP_ENDPOINT = f"http://{HOST}:{LLAMA_PORT}/completion"
LLAMA_CPP_HOME = "/opt/cx_intelligence/aiaas/compiled_llama_cpp"  # Replace with actual path to llama-server
LLAMA_SOURCE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "efs/frameworks/llama.cpp")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "efs/models/mistral-7b-instruct-v0.2.Q5_K_S.gguf")  # Replace with the actual path to your model
NUMBER_OF_CORES = multiprocessing.cpu_count()
WORKERS_IN_PARALLEL = f"-j {NUMBER_OF_CORES - 2}" if NUMBER_OF_CORES > 2 else ""

# In case of Windows Run add .exe extension
if PLATFORM == "Windows":
    LLAMA_CPP_PATH = os.path.join(LLAMA_CPP_HOME, "bin/Release/llama-server.exe")
else:
    LLAMA_CPP_PATH = os.path.join(LLAMA_CPP_HOME, "bin/Release/llama-server")


# Function to ensure llama.cpp repository exists
def ensure_llama_cpp_repository():
    # Check if Git repository exists and clone if it doesn't
    if not os.path.exists(os.path.join(LLAMA_SOURCE_FOLDER, ".git")):
        remove_directory(LLAMA_SOURCE_FOLDER)
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_SOURCE_FOLDER], check=True)


def remove_directory(dir_path):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory '{dir_path}' removed successfully.")
        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")


# Ensure llama.cpp repository exists
ensure_llama_cpp_repository()


def compile_llama_cpp():
    # Ensure LLAMA_CPP_HOME directory exists
    if not os.path.exists(LLAMA_CPP_HOME):
        # Change directory to LLAMA_CPP_HOME
        os.chdir(LLAMA_CPP_HOME)

        # Configure CMake
        gpu_support = "-DGGML_CUDA=ON"  # Adjust as needed based on your setup
        try:
            source_command = subprocess.run(
                ["cmake", "-B", ".", "-S", LLAMA_SOURCE_FOLDER, gpu_support],
                check=True,
                capture_output=True,
            )
            print(source_command.stdout.decode())
            print(source_command.stderr.decode())

            # Build llama-server target
            build_command = subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "--target", "llama-server", WORKERS_IN_PARALLEL],
                check=True,
                capture_output=True,
            )
            print(build_command.stdout.decode())
            print(build_command.stderr.decode())

        except subprocess.CalledProcessError as e:
            print(f"Error during cmake or build process: {e}")
            raise e


compile_llama_cpp()

# Global LLAMA_SERVER_PID variable
LLAMA_SERVER_PID = None


def start_llama_cpp():
    global LLAMA_SERVER_PID

    # Change working directory to LLAMA_CPP_HOME for starting llama-server
    os.chdir(LLAMA_CPP_HOME)

    llama_cpp_process = subprocess.Popen(
        [LLAMA_CPP_PATH, "--model", MODEL_PATH, "--ctx-size", CONTEXT_WINDOW, "--port", str(LLAMA_PORT), "-np", "2",
         "-ns", "1", "-ngl", "16", "-sm", "layer", "-ts", "0", "-mg", "-1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    LLAMA_SERVER_PID = llama_cpp_process.pid
    # Change back to the original working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


def stop_llama_cpp():
    global LLAMA_SERVER_PID
    command = f"taskkill /PID {LLAMA_SERVER_PID} /F" if PLATFORM == "Windows" else f"kill -9 {LLAMA_SERVER_PID}"
    subprocess.run(
        command.split()
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        stop_llama_cpp()


app = FastAPI(lifespan=lifespan)


@app.post("/infer")
def infer_text(text: str = Form()):
    payload = {
        "prompt": f"User: {text}\nAssistant:",
        "n_predict": -1,
        "temperature": 0.5  # Note the corrected temperature value
    }

    try:
        response = requests.post(LLAMA_CPP_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return JSONResponse(content={"result": result})
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with llama.cpp: {str(e)}")


# Capture SIGTERM signal to stop llama-server
def handle_sigterm(signum, frame):
    stop_llama_cpp()
    exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    start_llama_cpp()
    import uvicorn

    uvicorn.run(app, host=HOST, port=UVICORN_PORT)
