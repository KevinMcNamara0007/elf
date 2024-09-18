import multiprocessing
import os
import pickle
import platform
import subprocess
import shutil
import urllib.parse
from time import sleep

import asyncio

import psutil
from dotenv import load_dotenv
import onnxruntime as ort
import time
import requests
from fastapi import HTTPException

# Import ENV Vars
load_dotenv(os.getenv("ENV", "config/.env-dev"))
SPLIT_SYMBOL = os.getenv("SPLIT_SYMBOL")
general_model_path = os.getenv("general")
classifier_tokenizer = os.getenv("classifier_tokenizer")
classifier_model = os.getenv("classifier_model")
CONTEXT_WINDOW = os.getenv("CONTEXT_WINDOW")
INPUT_WINDOW = int(os.getenv("INPUT_WINDOW"))
HOST = os.getenv("HOST", "127.0.0.1")
UVICORN_PORT = os.getenv("UVICORN_PORT", "8000")
LLAMA_CPP_HOME = os.getenv("LLAMA_CPP_HOME", "/opt/cx_intelligence/aiaas/compiled_llama_cpp")
LLAMA_SOURCE_FOLDER = os.path.join(os.getcwd(), os.getenv("LLAMA_SOURCE_FOLDER"))
LLAMA_PORT = int(UVICORN_PORT) + 1
LLAMA_CPP_ENDPOINT = f"http://{HOST}:{LLAMA_PORT}/completion"
GENERAL_MODEL_PATH = os.path.join(os.getcwd(), general_model_path)
NUMBER_OF_CORES = multiprocessing.cpu_count()
WORKERS = f"-j {NUMBER_OF_CORES - 2}" if NUMBER_OF_CORES > 2 else ""
PLATFORM = platform.system()
GPU_LAYERS = os.getenv("GPU_LAYERS")
NUMBER_OF_SERVERS = int(os.getenv("NUMBER_OF_SERVERS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
UBATCH_SIZE = int(os.getenv("UBATCH_SIZE"))
API_TOKENS = os.getenv("API_TOKENS")
NO_TOKEN = "No Token was provided."
API_TOKENS = API_TOKENS.split(",")
LLAMA_CPP_ENDPOINTS = []
LLAMA_CPP_PATH = os.path.join(LLAMA_CPP_HOME, "bin/llama-server")
CHROMA_DATA_PATH = os.getenv("CHROMA_DATA_PATH")
CHROMA_PORT = NUMBER_OF_SERVERS + LLAMA_PORT
CHATML_TEMPLATE = os.getenv("CHATML_TEMPLATE")
LLAMA3_TEMPLATE = os.getenv("LLAMA3_TEMPLATE")
CHAT_TEMPLATE = LLAMA3_TEMPLATE if "LLAMA" in general_model_path.upper() else CHATML_TEMPLATE
MAX_RETRY_ATTEMPTS = 3
REQUEST_TIMEOUT = 300
RESTART_WAIT_TIME = 30
THREADS_BATCH = max(1, NUMBER_OF_CORES // 4)
RESTART_COMMAND = None
LLM_TIMEOUT = 75


def extract_port_from_url(url):
    parsed_url = urllib.parse.urlparse(url)
    port = parsed_url.port
    return port


def remove_directory(dir_path):
    """
    Remove a directory recursively
    """
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory '{dir_path}' removed successfully.")
        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")


def start_llama_cpp():
    """
    Orchestrates the start of llama-server with all of its requirements
    """
    # Change working directory to LLAMA_CPP_HOME for starting llama-server
    cwd = os.getcwd()
    os.chdir(LLAMA_CPP_HOME)
    # spin up x number of servers
    processes_ports = spin_up_server(number_of_servers=NUMBER_OF_SERVERS)
    # check health of all servers
    check_server_health(processes_ports)
    # Create a list of available servers
    for process_port in processes_ports:
        LLAMA_CPP_ENDPOINTS.append(f"http://{HOST}:{process_port[1]}")
    # Change back to the original working directory
    os.chdir(cwd)


def check_server_health(pids_ports):
    """
    Check the health of the llama-servers
    """
    if PLATFORM != "Windows":
        # Wait for the server to be ready
        max_attempts = 10
        wait_time = 2  # seconds

        for process_port in pids_ports:
            for attempt in range(max_attempts):
                time.sleep(wait_time)
                try:
                    # Use curl to check the server's /docs endpoint
                    curl_command = ["curl", "-s", f"http://localhost:{process_port[1]}/health"]
                    result = subprocess.run(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout = result.stdout.decode('utf-8')

                    # Check if the server is up and model is ready
                    if result.returncode == 0 and "Loading model" not in stdout:
                        print(
                            f"Connection to llama-server pid#{process_port[0].pid} on port#{process_port[1]} successful.")
                        break
                    elif result.returncode == 0 and "Loading model" in stdout:
                        print("Waiting for model to load...")
                        sleep(20)
                    else:
                        raise Exception(f"Connection attempt failed")

                except Exception as e:
                    print(
                        f"Attempt {attempt + 1}: Connection to llama-server on port http://localhost:{process_port[1]}/health failed. Retrying...")
                    if attempt == max_attempts - 1:
                        error_msg = process_port[0].communicate()[1].decode('utf-8')
                        print(f"Could not start llama-server.\n\nERROR:\n\n{error_msg}")
                        exit(1)


def spin_up_server(number_of_servers):
    """
    Starts up llama-server x (number_of_servers) with pre-configured params
    """
    global RESTART_COMMAND
    # Automatically adjust GPU layers, batch sizes, and thread settings based on server count
    gpu_layers = GPU_LAYERS if number_of_servers == 1 else str(int(GPU_LAYERS) // number_of_servers)
    batch_size = str(BATCH_SIZE) if number_of_servers == 1 else str(BATCH_SIZE // number_of_servers)
    ubatch_size = str(UBATCH_SIZE) if number_of_servers == 1 else str(UBATCH_SIZE // number_of_servers)
    threads = str(NUMBER_OF_CORES) if number_of_servers == 1 else str(NUMBER_OF_CORES // number_of_servers)
    threads_batch = str(THREADS_BATCH) if number_of_servers == 1 else str(THREADS_BATCH // number_of_servers)

    processes_ports = []
    for i in range(number_of_servers):
        PORT = LLAMA_PORT + i
        kill_process_on_port(PORT)
        # Form the command list dynamically
        command = [
            LLAMA_CPP_PATH,
            "--host", HOST,
            "--port", str(PORT),
            "--model", GENERAL_MODEL_PATH,
            "--ctx-size", str(CONTEXT_WINDOW),  # Specify context window size if needed
            "--gpu-layers", gpu_layers,  # Adjust GPU layers based on server count
            "--threads", threads,  # Adjust threads for CPU utilization
            "--threads-batch", threads_batch,  # Adjust threads batch dynamically
            "--batch-size", batch_size,  # Dynamically adjust batch size
            "--ubatch-size", ubatch_size  # Adjust micro-batch size
        ]
        RESTART_COMMAND = command
        processes_ports.append((subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE), PORT))
    return processes_ports


def file_cleanup(filename):
    """
    Removes files from given path
    """
    os.remove(filename)


def start_chroma_db(chroma_db_path=CHROMA_DATA_PATH):
    """
    Starts the ChromaDB server and verifies its startup.
    """
    cwd = os.getcwd()
    os.makedirs(chroma_db_path, exist_ok=True)
    os.chdir(chroma_db_path)

    # Command to start ChromaDB server
    command = [
        "chroma",
        "run",
        "--host", HOST,
        "--port", str(CHROMA_PORT),
    ]
    try:
        # Start the ChromaDB server
        CHROMA_SERVER_PID = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait and check server health using curl
        max_attempts = 5
        wait_time = 2  # seconds
        for attempt in range(max_attempts):
            time.sleep(wait_time)
            try:
                response = requests.get(f"http://{HOST}:{CHROMA_PORT}/health", timeout=3)
                if response.status_code == 200:
                    print(f"ChromaDB server started successfully on port {CHROMA_PORT}.")
                    return
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError):
                print(f"Attempt {attempt + 1}: Connection to ChromaDB server on port {CHROMA_PORT} failed. Retrying...")
                if attempt == max_attempts - 1:
                    error_msg = CHROMA_SERVER_PID.communicate()[1].decode('utf-8')
                    print(f"Could not start chromadb server after {max_attempts} attempts.\n\nERROR:\n\n{error_msg}")
                    exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start chromadb server: {e}")
        exit(1)
    finally:
        os.chdir(cwd)


async def restart_llama_server(port):
    """
    Restart the llama-server running on the specified port and wait for it to be ready.
    :param port: Port on which the llama-server is running.
    """
    global RESTART_COMMAND
    try:
        # rebuild restart command to adjust for potential port change
        port_index = RESTART_COMMAND.index("--port") + 1
        command = RESTART_COMMAND[:port_index] + [str(port)] + RESTART_COMMAND[port_index + 1:]

        # Kill any process using the specified port
        kill_process_on_port(port)

        # Wait for a bit to ensure the process is fully stopped
        await asyncio.sleep(5)

        # Restart the server with the provided command
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the server to start up
        await asyncio.sleep(RESTART_WAIT_TIME)

        print(f"llama-server restarted on port {port}")

    except Exception as e:
        print(f"Failed to restart llama-server on port {port}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to restart llama-server: {e}")


def kill_process_on_port(port, force_kill=True):
    """
    Kill any process using the specified port.
    :param port: The port to search for active processes.
    :param force_kill: If True, forcefully kill the process; otherwise, try to terminate first.
    """
    try:
        # Iterate through all network connections
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                # Get the process associated with this connection's PID
                process = psutil.Process(conn.pid)
                print(f"Found process {process.pid} ({process.name()}) using port {port}")

                # Try to terminate the process gracefully
                process.terminate()
                time.sleep(10)  # Wait a bit to ensure termination

                # If force_kill is True and process is still running, kill it
                if force_kill and process.is_running():
                    process.kill()
                    print(f"Force killed process {process.pid}")
                return

        print(f"No process found using port {port}")
    except Exception as e:
        print(f"Error killing process on port {port}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error killing process on port {port}: {e}")


# Load the tokenizer
with open(classifier_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the ONNX model
classifier = ort.InferenceSession(classifier_model, providers=['CoreMLExecutionProvider'])

# Available Classifications
classifications = {
    0: {"Model": GENERAL_MODEL_PATH, "Category": "code", "Link": LLAMA_CPP_ENDPOINTS},
    1: {"Model": GENERAL_MODEL_PATH, "Category": "language", "Link": LLAMA_CPP_ENDPOINTS},
    2: {"Model": GENERAL_MODEL_PATH, "Category": "math", "Link": LLAMA_CPP_ENDPOINTS},
}
