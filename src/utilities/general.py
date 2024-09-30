import multiprocessing
import os
import pickle
import platform
import signal
import subprocess
import shutil
import urllib.parse
from time import sleep
import asyncio
import psutil
from dotenv import load_dotenv
import onnxruntime as ort
import time
import random
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
CHROMA_PORT = LLAMA_PORT + 100  # Adjusted for dynamic server count
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
            "--ctx-size", "32000",
            "--port", str(PORT),
            "--model", GENERAL_MODEL_PATH,
            "--repeat-last-n", "0",
            "--gpu-layers", gpu_layers_per_server,
            "--threads", threads_per_server,
            "--threads-batch", threads_batch_per_server,
            "--batch-size", batch_size_per_server,
            "--ubatch-size", ubatch_size_per_server,
            "--penalize-nl",
            "--seed", "42",
            "--conversation"
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes_ports_commands.append((process, PORT, command))
    return processes_ports_commands


def file_cleanup(filename):
    os.remove(filename)


def start_chroma_db(chroma_db_path=CHROMA_DATA_PATH):
    kill_process_on_port(CHROMA_PORT)
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


def kill_process_on_port(port):
    """
    Finds and terminates the process running on the specified port.
    """
    for conn in psutil.net_connections(kind='inet'):
        # Check if the connection is listening or established on the specified port
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            # Get the process ID (pid) associated with the connection
            pid = conn.pid
            if pid is not None:
                try:
                    # Get the process by pid and terminate it gracefully
                    proc = psutil.Process(pid)
                    print(f"Terminating process {proc.name()} with PID {pid} on port {port}")
                    proc.terminate()  # Sends SIGTERM
                    proc.wait(timeout=3)  # Wait up to 3 seconds for the process to terminate
                except psutil.NoSuchProcess:
                    print(f"No process found with PID {pid}")
                except psutil.TimeoutExpired:
                    print(f"Process {pid} did not terminate in time, forcing a kill.")
                    proc.kill()  # Sends SIGKILL if termination fails
                except Exception as e:
                    print(f"Error terminating process on port {port}: {e}")


async def kill_process_on_port_async(port):
    """
    Finds and terminates the process running on the specified port.
    """
    for conn in psutil.net_connections(kind='inet'):
        # Check if the connection is listening or established on the specified port
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            # Get the process ID (pid) associated with the connection
            pid = conn.pid
            if pid is not None:
                try:
                    # Get the process by pid and terminate it gracefully
                    proc = psutil.Process(pid)
                    print(f"Terminating process {proc.name()} with PID {pid} on port {port}")
                    proc.terminate()  # Sends SIGTERM
                    proc.wait(timeout=3)  # Wait up to 3 seconds for the process to terminate
                except psutil.NoSuchProcess:
                    print(f"No process found with PID {pid}")
                except psutil.TimeoutExpired:
                    print(f"Process {pid} did not terminate in time, forcing a kill.")
                    proc.kill()  # Sends SIGKILL if termination fails
                except Exception as e:
                    print(f"Error terminating process on port {port}: {e}")


async def restart_llama_server(port, command):
    try:
        await kill_process_on_port_async(port)  # Ensure the old process is killed asynchronously

        # Restart llama server using asyncio to prevent blocking
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print(f"llama-server restarted on port {port}")
        return process
    except Exception as e:
        print(f"Failed to restart llama-server on port {port}: {str(e)}")
        raise Exception(f"Failed to restart llama-server: {e}")


class Server:
    def __init__(self, process, port):
        self.process = process
        self.port = port
        self.call_count = 0
        self.max_calls = random.randint(25, 35)  # Generate random max calls


class ServerManager:
    def __init__(self, number_of_servers):
        servers_pids_ports_commands = spin_up_server(number_of_servers)
        self.servers = [Server(process, port) for process, port, _ in servers_pids_ports_commands]
        self.standby_server = self.servers.pop()  # Remove the last server as the standby
        self.active_servers = self.servers  # All remaining servers are active
        self.lock = asyncio.Lock()  # Async lock for safe access
        self.current_server_index = 0  # Track the current active server index for round-robin

    async def increment_call_count(self, server_index):
        async with self.lock:
            server = self.active_servers[server_index]
            server.call_count += 1
            print(f"Active server on port {server.port} has handled {server.call_count} calls.")

            # Check if the server has reached its max call count
            if server.call_count >= server.max_calls:
                print(f"Server on port {server.port} reached max calls. Swapping with standby server.")
                asyncio.create_task(self.swap_server(server_index))  # Handle the swap in the background

    async def swap_server(self, server_index):
        async with self.lock:
            active_server = self.active_servers[server_index]
            print(
                f"Swapping active server on port {active_server.port} with standby server on port {self.standby_server.port}.")

            # Promote the standby server to active
            self.active_servers[server_index] = self.standby_server

            # Reset call count and max_calls for the new active server
            self.standby_server.call_count = 0
            self.standby_server.max_calls = random.randint(25, 35)
            print(
                f"New active server on port {self.standby_server.port} will handle {self.standby_server.max_calls} calls.")

            # Restart the old active server as the new standby in the background
            old_standby = active_server  # Save reference to the old active server

            # Restart the standby server asynchronously
            asyncio.create_task(self.restart_standby(old_standby))

            # Set the old active server as the new standby
            self.standby_server = old_standby

    async def restart_standby(self, server):
        print(f"Restarting server on port {server.port} in the background...")
        # Ensure you're using an async-friendly subprocess call to restart the server
        server.process = await restart_llama_server(server.port, server.process.args)
        print(f"Server on port {server.port} restarted and ready as standby.")

    def get_current_server_endpoint(self):
        # Return the endpoint of the active server in a round-robin manner
        self.current_server_index = (self.current_server_index + 1) % len(
            self.active_servers)  # Move to the next server
        return f"http://localhost:{self.active_servers[self.current_server_index].port}"


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
