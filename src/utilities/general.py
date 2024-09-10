import multiprocessing
import os
import pickle
import platform
import subprocess
import shutil
from dotenv import load_dotenv
import onnxruntime as ort
from time import sleep
import requests


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
        for process_port in pids_ports:
            for attempt in range(max_attempts):
                try:
                    # Use curl to check the server's /docs endpoint
                    curl_command = ["curl", "-s", f"http://localhost:{process_port[1]}/health"]
                    result = subprocess.run(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout = result.stdout.decode('utf-8')

                    # Check if the server is up and model is ready
                    if result.returncode == 0 and "Loading model" not in stdout:
                        print(f"Connection to llama-server pid#{process_port[0].pid} on port#{process_port[1]} successful.")
                        break
                    elif result.returncode == 0 and "Loading model" in stdout:
                        print("Waiting for model to load...")
                        sleep(30)
                    else:
                        raise Exception(f"Connection attempt failed")

                except Exception as e:
                    print(f"Attempt {attempt + 1}: Connection to llama-server on port http://localhost:{process_port[1]}/health failed {str(e)}.\n Retrying...")
                    if attempt == max_attempts - 1:
                        error_msg = process_port[0].communicate()[1].decode('utf-8')
                        print(f"Could not start llama-server.\n\nERROR:\n\n{error_msg}")
                        exit(1)


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
            "--threads", str(NUMBER_OF_CORES) if number_of_servers == 1 else str(NUMBER_OF_CORES // number_of_servers),
            # Allowable threads for CPU operations
            "--batch-size", str(BATCH_SIZE) if number_of_servers == 1 else str(BATCH_SIZE // number_of_servers),
            # logical maximum batch size (default: 2048)
            "--ubatch-size", str(UBATCH_SIZE) if number_of_servers == 1 else str(UBATCH_SIZE // number_of_servers),
            # physical maximum batch size (default: 512)
        ]
        processes_ports.append((subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE), PORT))
    return processes_ports


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
            sleep(wait_time)
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
