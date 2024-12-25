import os
import platform
import time
import httpx
import psutil
from dotenv import load_dotenv

from src.modeling.classifier_manager import ClassifierManager
from src.modeling.llama_server_manager import LlamaServerManager

# Import ENV Vars
load_dotenv("config/.env-dev")
SPLIT_SYMBOL = os.getenv("SPLIT_SYMBOL", ":::")
classifier_tokenizer = os.getenv("classifier_tokenizer", "efs/classifier/tokenizer.pickle")
classifier_model = os.getenv("classifier_model", "efs/classifer/KM_Classifier.onnx")
LLAMA_PORT = os.getenv("LLAMA_PORT", "8001")
PLATFORM = platform.system()
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "99"))
API_TOKENS = os.getenv("API_TOKENS", "nfaiwune2iun3onrofni2n3o1n,134921ur81298n2309n")
NO_TOKEN = "No Token was provided."
API_TOKENS = API_TOKENS.split(",")
CHROMA_DATA_PATH = os.getenv("CHROMA_DATA_PATH", "efs/chroma_data")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8099"))

llama_manager = LlamaServerManager()
classifier_manager = ClassifierManager()

# CNN classes:
classifications = ['code', 'general', 'math']

STREAM_PAYLOAD = {
    "stream": True,  # Enable streaming
    "temperature": 0.8,
    "repeat_last_n": 0,
    "repeat_penalty": 1,
    "penalize_nl": True,
    "top_k": 0,
    "top_p": 1,
    "min_p": 0.05,
    "tfs_z": 1,
    "typical_p": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "mirostat": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "grammar": "",
    "n_probs": 0,
    "min_keep": 0,
    "image_data": [],
    "cache_prompt": False,
    "api_key": ""}


# General Helper Functions
def kill_process_on_port(port):
    """
    Kills a process on a given port.
    """
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            pid = conn.pid
            if pid:
                try:
                    os.kill(pid, 9)
                    print(f"Process {pid} on port {port} has been killed.")
                except Exception as e:
                    print(f"Failed to kill process {pid} on port {port}: {e}")
            else:
                print(f"No process found running on port {port}.")
            return
    print(f"No process found running on port {port}.")


async def check_server_health(port, max_attempts=10, endpoint="/health"):
    """
    Check the health of a llama-server on a specific port.
    """
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"http://localhost:{port}{endpoint}")
                if response.status_code == 200 and response.json().get("status") == "ok":
                    print(f"Server on port {port} is healthy.")
                    return True
                else:
                    print(f"Server on port {port} is not ready yet, retrying...")
                    time.sleep(5)
        except Exception as e:
            print(f"Health check failed for server on port {port}: {e}. Retrying...")
            time.sleep(2)

    print(f"Failed to start server on port {port}. Max attempts reached.")
    return False


async def start_aux_servers():
    """
    Starts auxiliary servers for ChromaDB, llama, and classifier managers.
    Returns the initialized server managers.
    """
    global llama_manager, classifier_manager

    # Spin up the servers
    await llama_manager.spin_up_servers()
    classifier_manager.start_classifier()

    # # Check health for ChromaDB server
    # chroma_manager.start_chroma_db()


def stop_aux_servers():
    """
    Shuts down ChromaDB, llama servers, and classifier by killing the processes on known ports.
    """
    # Get the number of servers and ports from environment variables
    llama_port = int(os.getenv("LLAMA_PORT", 8001))
    number_of_servers = int(os.getenv('NUMBER_OF_SERVERS', 1))

    # Kill llama servers
    for i in range(number_of_servers):
        port_to_kill = llama_port + i
        kill_process_on_port(port_to_kill)
        print(f"Killed llama-server on port {port_to_kill}")
