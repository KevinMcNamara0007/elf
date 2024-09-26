import multiprocessing
import os
import pickle
import platform
import subprocess
import shutil
import urllib.parse
from time import sleep
import asyncio
import threading
import psutil
from dotenv import load_dotenv
import onnxruntime as ort
import time
import requests
from fastapi import HTTPException

# Import ENV Vars
load_dotenv(os.getenv("ENV", "config/.env-dev"))
SPLIT_SYMBOL = os.getenv("SPLIT_SYMBOL")
general_model_path = os.getenv("general", "efs/models/Llama-3.1.gguf")
classifier_tokenizer = os.getenv("classifier_tokenizer")
classifier_model = os.getenv("classifier_model")
HOST = os.getenv("HOST", "127.0.0.1")
UVICORN_PORT = os.getenv("UVICORN_PORT", "8000")
LLAMA_CPP_HOME = os.getenv("LLAMA_CPP_HOME", "/opt/cx_intelligence/aiaas/compiled_llama_cpp")
LLAMA_SOURCE_FOLDER = os.path.join(os.getcwd(), os.getenv("LLAMA_SOURCE_FOLDER"))
LLAMA_PORT = int(UVICORN_PORT) + 1
LLAMA_CPP_ENDPOINT = f"http://{HOST}:{LLAMA_PORT}/completion"
GENERAL_MODEL_PATH = os.path.join(os.getcwd(), general_model_path)
NUMBER_OF_SERVERS = os.getenv("NUMBER_OF_SERVERS", "1")
NUMBER_OF_CORES = multiprocessing.cpu_count()
PLATFORM = platform.system()
GPU_LAYERS = int(os.getenv("GPU_LAYERS"))
TOTAL_BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
TOTAL_UBATCH_SIZE = int(os.getenv("UBATCH_SIZE"))
API_TOKENS = os.getenv("API_TOKENS")
NO_TOKEN = "No Token was provided."
API_TOKENS = API_TOKENS.split(",")
LLAMA_CPP_ENDPOINTS = []
LLAMA_CPP_PATH = os.path.join(LLAMA_CPP_HOME, "bin/llama-server")
CHROMA_DATA_PATH = os.getenv("CHROMA_DATA_PATH")
CHROMA_PORT = LLAMA_PORT + int(os.getenv("NUMBER_OF_SERVERS")) + 1  # Adjusted for dynamic server count
CHATML_TEMPLATE = os.getenv("CHATML_TEMPLATE")
LLAMA3_TEMPLATE = os.getenv("LLAMA3_TEMPLATE")
CHAT_TEMPLATE = LLAMA3_TEMPLATE if "LLAMA" in general_model_path.upper() else CHATML_TEMPLATE
MAX_RETRY_ATTEMPTS = 3
REQUEST_TIMEOUT = 300
RESTART_WAIT_TIME = 10
TOTAL_THREADS_BATCH = max(1, NUMBER_OF_CORES // 4)
LLM_TIMEOUT = 75
SERVER_MANAGER = None


def extract_port_from_url(url):
    parsed_url = urllib.parse.urlparse(url)
    port = parsed_url.port
    return port


def remove_directory(dir_path):
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory '{dir_path}' removed successfully.")
        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")


def start_llama_cpp():
    global SERVER_MANAGER
    check_or_find_gguf_file()
    cwd = os.getcwd()
    os.chdir(LLAMA_CPP_HOME)
    number_of_servers = int(os.getenv("NUMBER_OF_SERVERS"))
    server_manager = ServerManager(number_of_servers)
    processes_ports = [(server.process, server.port) for server in
                       server_manager.servers + [server_manager.standby_server]]
    check_server_health(processes_ports)
    LLAMA_CPP_ENDPOINTS.clear()
    for server in server_manager.servers:
        LLAMA_CPP_ENDPOINTS.append(f"http://{HOST}:{server.port}")
    os.chdir(cwd)
    SERVER_MANAGER = server_manager


def check_or_find_gguf_file():
    global GENERAL_MODEL_PATH
    if os.path.exists(GENERAL_MODEL_PATH):
        print(f"File exists: {GENERAL_MODEL_PATH}")
    else:
        print(f"File does not exist: {GENERAL_MODEL_PATH}")
        directory = os.path.dirname(GENERAL_MODEL_PATH)
        gguf_files = [f for f in os.listdir(directory) if f.endswith('.gguf')]
        if gguf_files:
            print(f"Found .gguf file: {gguf_files}")
            GENERAL_MODEL_PATH = os.path.join(os.getcwd(), "efs", "models", gguf_files[0])
        else:
            raise FileNotFoundError(f"No .gguf file found in directory: {directory}")


def check_server_health(pids_ports):
    if PLATFORM != "Windows":
        max_attempts = 10
        wait_time = 2  # seconds

        for process_port in pids_ports:
            for attempt in range(max_attempts):
                time.sleep(wait_time)
                try:
                    curl_command = ["curl", "-s", f"http://localhost:{process_port[1]}/health"]
                    result = subprocess.run(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout = result.stdout.decode('utf-8')
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
    total_servers = int(number_of_servers) + 1
    gpu_layers_per_server = str(max(1, GPU_LAYERS // total_servers))
    batch_size_per_server = str(max(1, TOTAL_BATCH_SIZE // total_servers))
    ubatch_size_per_server = str(max(1, TOTAL_UBATCH_SIZE // total_servers))
    threads_per_server = str(max(1, NUMBER_OF_CORES // total_servers))
    threads_batch_per_server = str(max(1, TOTAL_THREADS_BATCH // total_servers))

    processes_ports_commands = []
    for i in range(total_servers):
        PORT = LLAMA_PORT + i
        kill_process_on_port(PORT)
        command = [
            LLAMA_CPP_PATH,
            "--host", HOST,
            "--port", str(PORT),
            "--model", GENERAL_MODEL_PATH,
            "--repeat-last-n", "0",
            "--gpu-layers", gpu_layers_per_server,
            "--threads", threads_per_server,
            "--threads-batch", threads_batch_per_server,
            "--batch-size", batch_size_per_server,
            "--ubatch-size", ubatch_size_per_server,
            "--dump-kv-cache",
            "--penalize-nl",
            "--seed", "42"
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes_ports_commands.append((process, PORT, command))
    return processes_ports_commands


def file_cleanup(filename):
    os.remove(filename)


def start_chroma_db(chroma_db_path=CHROMA_DATA_PATH):
    cwd = os.getcwd()
    os.makedirs(chroma_db_path, exist_ok=True)
    os.chdir(chroma_db_path)
    command = [
        "chroma",
        "run",
        "--host", HOST,
        "--port", str(CHROMA_PORT),
    ]
    try:
        CHROMA_SERVER_PID = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


async def restart_llama_server(port, command):
    try:
        kill_process_on_port(port)
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        await asyncio.sleep(RESTART_WAIT_TIME)
        print(f"llama-server restarted on port {port}")
    except Exception as e:
        print(f"Failed to restart llama-server on port {port}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to restart llama-server: {e}")


def kill_process_on_port(port, force_kill=True):
    try:
        process = psutil.Process()
        for connection in process.connections(kind='inet'):
            if connection.laddr.port == port:
                pid = connection.pid
                if force_kill:
                    os.kill(pid, 9)
                else:
                    os.kill(pid, 15)
                print(f"Killed process running on port {port}.")
    except Exception as e:
        print(f"Failed to kill process on port {port}: {str(e)}")


import random


class Server:
    def __init__(self, process, port):
        self.process = process
        self.port = port
        self.call_count = 0
        self.max_calls = random.randint(25, 35)  # Generate random max calls


class ServerManager:
    def __init__(self, number_of_servers):
        servers_pids_ports_commands = spin_up_server(number_of_servers)
        self.servers = [Server(process, port) for process, port, _ in servers_pids_ports_commands[:-1]]
        standby_process, standby_port, standby_command = servers_pids_ports_commands[-1]
        self.standby_server = Server(standby_process, standby_port)
        self.standby_command = standby_command
        self.lock = threading.Lock()

    def increment_call_count(self, server_index):
        server = self.servers[server_index]
        server.call_count += 1

        # Check if call count exceeds max_calls
        if server.call_count >= server.max_calls:
            print(f"Server on port {server.port} has reached {server.call_count} calls and will be swapped.")
            asyncio.run(self.swap_and_restart_server(server_index))

    async def swap_and_restart_server(self, server_index):
        with self.lock:
            server = self.servers[server_index]
            standby = self.standby_server

            # Swap the server with the standby server
            self.servers[server_index], self.standby_server = standby, server
            print(f"Swapping server on port {server.port} with standby server on port {standby.port}.")

            # Reset call count and assign new max_calls for the newly active server
            self.servers[server_index].call_count = 0
            self.servers[server_index].max_calls = random.randint(25, 35)
            print(
                f"New server on port {self.servers[server_index].port} will handle {self.servers[server_index].max_calls} calls.")

            # Restart the old server (now standby)
            await restart_llama_server(server.port, self.standby_command)

    def handle_request(self, server_index):
        # Handle request logic (e.g., processing API calls, inference, etc.)
        self.increment_call_count(server_index)


# Load the tokenizer
with open(classifier_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the ONNX model
classifier = ort.InferenceSession(
    classifier_model,
    providers=['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider'])

# Available Classifications
classifications = {
    0: {"Model": GENERAL_MODEL_PATH, "Category": "code", "Link": LLAMA_CPP_ENDPOINTS},
    1: {"Model": GENERAL_MODEL_PATH, "Category": "language", "Link": LLAMA_CPP_ENDPOINTS},
    2: {"Model": GENERAL_MODEL_PATH, "Category": "math", "Link": LLAMA_CPP_ENDPOINTS},
}
