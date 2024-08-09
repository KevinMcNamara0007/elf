import multiprocessing
import os
import pickle
import platform
import subprocess
import shutil
from dotenv import load_dotenv
import onnxruntime as ort
import time
import requests
import psutil
from transformers import BertTokenizer

# Global LLAMA_SERVER_PID variable
LLAMA_SERVER_PID = None

CHROMA_SERVER_PID = None

# Import ENV Vars
load_dotenv(os.getenv("ENV", "config/.env-dev"))
general_model_path = os.getenv("general")
programming_model_path = os.getenv("programming")
classifier_tokenizer = os.getenv("classifier_tokenizer")
classifier_model = os.getenv("classifier_model")
stt_model_path = os.getenv("stt_model_path")
vision_model_path = os.getenv("vision_model_path")
CONTEXT_WINDOW = os.getenv("CONTEXT_WINDOW")
INPUT_WINDOW = int(os.getenv("INPUT_WINDOW"))
HOST = os.getenv("HOST")
UVICORN_PORT = os.getenv("UVICORN_PORT")
LLAMA_CPP_HOME = os.getenv("LLAMA_CPP_HOME")
LLAMA_SOURCE_FOLDER = os.path.join(os.getcwd(), os.getenv("LLAMA_SOURCE_FOLDER"))
LLAMA_PORT = int(UVICORN_PORT) + 1
LLAMA_CPP_ENDPOINT = f"http://{HOST}:{LLAMA_PORT}/completion"
MAIN_GPU_INDEX = os.getenv("MAIN_GPU_INDEX")
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
LLAMA_CPP_PATH = os.path.join(LLAMA_CPP_HOME, "bin/Release/llama-server")
CHROMA_FILE_PATH = os.getenv("CHROMA_FILE_PATH")
CHROMA_PORT = NUMBER_OF_SERVERS + LLAMA_PORT
ONNX_Embedding_MODEL_PATH = os.getenv("ONNX_Embedding_MODEL_PATH")
ONNX_Embedding_TOKENIZER_PATH = os.getenv("ONNX_Embedding_TOKENIZER_PATH")

embedding_tokenizer = BertTokenizer.from_pretrained(ONNX_Embedding_TOKENIZER_PATH)
embedding_model = ort.InferenceSession(ONNX_Embedding_MODEL_PATH)


# Function to ensure llama.cpp repository exists
def ensure_llama_cpp_repository():
    """
    Check if Git repository exists and clone if it doesn't
    """
    if not os.path.exists(os.path.join(LLAMA_SOURCE_FOLDER, ".git")):
        remove_directory(LLAMA_SOURCE_FOLDER)
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_SOURCE_FOLDER], check=True)


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


def compile_llama_cpp():
    """
    Compiles the llama-server using cmake and GPU acceleration (if available)
    """
    global LLAMA_CPP_PATH
    LLAMA_CPP_PATH = check_possible_paths()
    # Ensure LLAMA_CPP_HOME directory exists
    if LLAMA_CPP_PATH == "":
        # Change directory to LLAMA_CPP_HOME
        remove_directory(LLAMA_CPP_HOME)
        os.makedirs(LLAMA_CPP_HOME, exist_ok=True)
        os.chdir(LLAMA_CPP_HOME)

        # Configure CMake
        gpu_support = "-DGGML_CUDA=ON" if platform.system() != "Darwin" else ""  # Adjust as needed based on your setup
        try:
            source_command = subprocess.run(
                ["cmake", "..", "-B", ".", "-S", LLAMA_SOURCE_FOLDER, gpu_support],
                check=True,
                capture_output=True,
            )
            print(source_command.stdout.decode())
            print(source_command.stderr.decode())

            # Build llama-server target
            build_command = subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "--target", "llama-server", WORKERS],
                check=True,
                capture_output=True,
            )
            print(build_command.stdout.decode())
            print(build_command.stderr.decode())
            LLAMA_CPP_PATH = check_possible_paths()
            if LLAMA_CPP_PATH == "":
                raise ConnectionError("Could not find llama-server.")
        except subprocess.CalledProcessError as e:
            print(f"Error during cmake or build process: {e}")
            raise e


def check_possible_paths():
    """
    Check the possible paths of the llama-server
    Inconsistencies across platforms are rectified in this method.
    """
    global LLAMA_CPP_PATH
    if not os.path.exists(LLAMA_CPP_PATH):
        split_path = LLAMA_CPP_PATH.split("Release/")
        LLAMA_CPP_PATH = os.path.join(split_path[0], split_path[1])
        if not os.path.exists(LLAMA_CPP_PATH):
            return ""
    return LLAMA_CPP_PATH


def start_llama_cpp():
    """
    Orchestrates the start of llama-server with all of its requirements
    """
    global LLAMA_SERVER_PID
    # Ensure llama.cpp repository exists
    ensure_llama_cpp_repository()
    for i in range(NUMBER_OF_SERVERS):
        kill_process_on_port(LLAMA_PORT + i)
    # Compile the folder
    compile_llama_cpp()
    # Change working directory to LLAMA_CPP_HOME for starting llama-server
    cwd = os.getcwd()
    os.chdir(LLAMA_CPP_HOME)
    # spin up x number of servers
    processes_ports = spin_up_server(number_of_servers=NUMBER_OF_SERVERS)
    # check health of all servers
    check_server_health(processes_ports)
    LLAMA_SERVER_PID = processes_ports
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
        max_attempts = 5
        wait_time = 2  # seconds
        for process_port in pids_ports:
            for attempt in range(max_attempts):
                time.sleep(wait_time)
                try:
                    response = requests.get(f"http://{HOST}:{process_port[1]}/docs", timeout=3)
                    if response.status_code == 200:
                        print(
                            f"Connection to llama-server pid#{process_port[0].pid} on port#{process_port[1]} successful.")
                        break
                except (
                        requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                        requests.exceptions.HTTPError) as e:
                    print(
                        f"Attempt {attempt + 1}: Connection to llama-server on port http://{HOST}:{process_port[1]}/health failed. Retrying...")
                    if attempt == max_attempts - 1:
                        print(Exception(
                            f"Could not start llama-server.\n\nERROR:\n\n{process_port[0].communicate()[1].decode('utf-8')}"))
                        exit(1)
                    continue


def spin_up_server(number_of_servers):
    """
    Starts up llama-server x (number_of_servers) with pre-configured params
    """
    processes_ports = []
    for i in range(number_of_servers):
        PORT = LLAMA_PORT + i
        command = [
            LLAMA_CPP_PATH,
            "--host", HOST,
            "--port", str(PORT),
            "--model", GENERAL_MODEL_PATH,
            "--ctx-size", CONTEXT_WINDOW,
            "--gpu-layers", GPU_LAYERS if number_of_servers == 1 else str(int(GPU_LAYERS) // number_of_servers),
            # Number of GPU layers
            "--threads", str(NUMBER_OF_CORES) if number_of_servers == 1 else str(NUMBER_OF_CORES // number_of_servers),
            # Allowable threads for CPU operations
            "--batch-size", str(BATCH_SIZE) if number_of_servers == 1 else str(BATCH_SIZE // number_of_servers),
            # logical maximum batch size (default: 2048)
            "--ubatch-size", str(UBATCH_SIZE) if number_of_servers == 1 else str(UBATCH_SIZE // number_of_servers),
            # physical maximum batch size (default: 512)
            "--conversation",
        ]
        processes_ports.append((subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE), PORT))
    return processes_ports


def stop_aux_servers():
    """
    Kills all llama-servers
    """
    global LLAMA_SERVER_PID
    global CHROMA_SERVER_PID
    if LLAMA_SERVER_PID:
        for pid_port in LLAMA_SERVER_PID:
            kill_process_on_port(pid_port[1])
        LLAMA_SERVER_PID = None
    if CHROMA_SERVER_PID:
        kill_process_on_port(CHROMA_PORT)
        CHROMA_SERVER_PID = None


def kill_process_on_port(port):
    """
    Kills a process on a given port
    """
    # Iterate over all the network connections
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            pid = conn.pid
            if pid:
                try:
                    # Kill the process
                    os.kill(pid, 9)
                    print(f"Process {pid} on port {port} has been killed.")
                except Exception as e:
                    print(f"Failed to kill process {pid} on port {port}: {e}")
            else:
                print(f"No process found running on port {port}.")
            return

    print(f"No process found running on port {port}.")


# Capture SIGTERM signal to stop llama-server
def handle_sigterm(signum, frame):
    """
    Listens for SIGTERM and kills llama-servers
    """
    kill_process_on_port(CHROMA_PORT)
    stop_aux_servers()
    exit(0)


def replace_port_in_url(url, new_port):
    """
    Replaces port in a given url with new port
    """
    parts = url.split(":")
    parts[-1] = str(new_port) + "/" + parts[-1].split("/")[-1]  # Replace the second last part (the port)
    return ":".join(parts)


def file_cleanup(filename):
    """
    Removes files from given path
    """
    os.remove(filename)


def start_chroma_db(chroma_db_path=CHROMA_FILE_PATH):
    """
    Starts the ChromaDB server and verifies its startup.
    """
    global CHROMA_SERVER_PID
    os.makedirs(chroma_db_path, exist_ok=True)

    # Kill any existing process on the ChromaDB port
    kill_process_on_port(CHROMA_PORT)

    # Command to start ChromaDB server
    command = [
        "chroma",
        "run",
        "--host", HOST,
        "--port", str(CHROMA_PORT),
        "--path", chroma_db_path
    ]

    try:
        # Start the ChromaDB server
        CHROMA_SERVER_PID = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait and check server health
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
                    print(f"Could not start ChromaDB server.\n\nERROR:\n\n{error_msg}")
                    exit(1)

    except subprocess.CalledProcessError as e:
        print(f"Failed to start ChromaDB server: {e}")
        exit(1)


# Load the tokenizer
with open(classifier_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the ONNX model
classifier = ort.InferenceSession(classifier_model)

# Available Classifications
classifications = {
    0: {"Model": GENERAL_MODEL_PATH, "Category": "code", "Link": LLAMA_CPP_ENDPOINTS},
    1: {"Model": GENERAL_MODEL_PATH, "Category": "language", "Link": LLAMA_CPP_ENDPOINTS},
    2: {"Model": GENERAL_MODEL_PATH, "Category": "math", "Link": LLAMA_CPP_ENDPOINTS},
}
