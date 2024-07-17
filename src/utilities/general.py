import multiprocessing
import os
import pickle
import platform
import subprocess
import shutil
from dotenv import load_dotenv
import onnxruntime as ort
import time
import socket
import requests
import psutil

# Global LLAMA_SERVER_PID variable
LLAMA_SERVER_PID = None


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
GENERAL_MODEL_PATH = os.path.join(os.getcwd(), general_model_path)
NUMBER_OF_CORES = multiprocessing.cpu_count()
WORKERS = f"-j {NUMBER_OF_CORES - 2}" if NUMBER_OF_CORES > 2 else ""
PLATFORM = platform.system()

stt_model_id = stt_model_path if os.path.exists(stt_model_path) else "openai/whisper-medium"

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


def compile_llama_cpp():
    global LLAMA_CPP_PATH
    LLAMA_CPP_PATH = check_possible_paths()
    # Ensure LLAMA_CPP_HOME directory exists
    if LLAMA_CPP_PATH == "":
        # Change directory to LLAMA_CPP_HOME
        remove_directory(LLAMA_CPP_HOME)
        os.makedirs(LLAMA_CPP_HOME, exist_ok=True)
        os.chdir(LLAMA_CPP_HOME)

        # Configure CMake
        gpu_support = "-DGGML_CUDA=ON" if platform.system() != "Darwin" else "-DLLAMA_USE_METAL=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++" # Adjust as needed based on your setup
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
                raise RuntimeException("Could not find llama-server.")
        except subprocess.CalledProcessError as e:
            print(f"Error during cmake or build process: {e}")
            raise e


def check_possible_paths():
    global LLAMA_CPP_PATH
    if not os.path.exists(LLAMA_CPP_PATH):
        split_path = LLAMA_CPP_PATH.split("Release/")
        LLAMA_CPP_PATH = os.path.join(split_path[0], split_path[1])
        if not os.path.exists(LLAMA_CPP_PATH):
            return ""
    return LLAMA_CPP_PATH


def start_llama_cpp():
    # Ensure llama.cpp repository exists
    ensure_llama_cpp_repository()
    kill_process_on_port(LLAMA_PORT)
    # Compile the folder
    compile_llama_cpp()
    global LLAMA_SERVER_PID


    # Change working directory to LLAMA_CPP_HOME for starting llama-server
    cwd = os.getcwd()
    os.chdir(LLAMA_CPP_HOME)
    command = [
        LLAMA_CPP_PATH,
        "--model", GENERAL_MODEL_PATH,
        "--ctx-size", CONTEXT_WINDOW,
        "--port", str(LLAMA_PORT),
        "--host", HOST,
        "-sm", "layer",
        "-ngl", "-1",  # Reduced number of GPU layers
        "-ts", "0",  # Tensor split
        "-mg", "-1",  # Main gpu,
    ]
    # print(' '.join(command))
    llama_cpp_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    LLAMA_SERVER_PID = llama_cpp_process.pid
    print("LLAMA_SERVER_PID:", LLAMA_SERVER_PID)

    # Wait for the server to be ready
    max_attempts = 5
    wait_time = 2  # seconds

    if PLATFORM != "Windows":
        for attempt in range(max_attempts):
            time.sleep(wait_time)
            try:
                response = requests.get(f"http://{HOST}:{LLAMA_PORT}/health", timeout=3)
                if response.status_code == 200:
                    print(f"Connection to llama-server on port {LLAMA_PORT} successful.")
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                print(f"Attempt {attempt + 1}: Connection to llama-server on port http://{HOST}:{LLAMA_PORT}/health failed. Retrying...")
                if attempt == max_attempts - 1:
                    print(Exception(
                        f"Could not start llama-server.\n\nERROR:\n\n{llama_cpp_process.communicate()[1].decode('utf-8')}"))
                    exit(1)
                continue
        # You may choose to handle this error, raise an exception, or take other actions.
    # Change back to the original working directory
    os.chdir(cwd)


def stop_llama_cpp():
    global LLAMA_SERVER_PID
    if LLAMA_SERVER_PID:
        kill_process_on_port(LLAMA_PORT)
        LLAMA_SERVER_PID = None

def kill_process_on_port(port):
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
    stop_llama_cpp()
    exit(0)


def replace_port_in_url(url, new_port):
    parts = url.split(":")
    parts[-1] = str(new_port) + "/" + parts[-1].split("/")[-1]  # Replace the second last part (the port)
    return ":".join(parts)


# Load the tokenizer
with open(classifier_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the ONNX model
classifier = ort.InferenceSession(classifier_model)

# Available Classifications
classifications = {
    0: {"Model": GENERAL_MODEL_PATH, "Category": "code", "Link": replace_port_in_url(LLAMA_CPP_ENDPOINT, LLAMA_PORT)},
    1: {"Model": GENERAL_MODEL_PATH, "Category": "language",
        "Link": replace_port_in_url(LLAMA_CPP_ENDPOINT, LLAMA_PORT)},
    2: {"Model": GENERAL_MODEL_PATH, "Category": "math", "Link": replace_port_in_url(LLAMA_CPP_ENDPOINT, LLAMA_PORT)}
}


# programming_expert = Llama(model_path=programming_model_path, n_gpu_layers=-1, n_ctx=2048)

# # Load vision model
# vision_model_id = vision_model_path if os.path.exists(vision_model_path) else "microsoft/Phi-3-vision-128k-instruct"
#
# vision_model = AutoModelForCausalLM.from_pretrained(
#     vision_model_id,
#     device_map=device,
#     trust_remote_code=True,
#     torch_dtype="auto",
#     _attn_implementation="eager"  # use _attn_implementation='flash_attention_2' to enable flash attention
# )
#
# vision_processor = AutoProcessor.from_pretrained(
#     vision_model_id,
#     trust_remote_code=True
# )
#
# if not os.path.exists(vision_model_path):
#     vision_model.save_pretrained(
#         vision_model_path,
#         is_main_process=True,
#         save_functional=True,
#         save_classif_head=True,
#         save_tokenizer=True,
#         save_shared=True,  # Ensure shared tensors are saved
#         safe_serialization=False  # Bypass safety check for shared tensors
#     )
#     vision_processor.save_pretrained(
#         vision_model_path,
#         is_main_process=True,
#         save_functional=True,
#         save_classif_head=True,
#         save_tokenizer=True,
#         save_shared=True,  # Ensure shared tensors are saved
#         safe_serialization=False  # Bypass safety check for shared tensors
#     )

# # Load Speech to Text Model
# stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     stt_model_id,
#     low_cpu_mem_usage=True,
#     use_safetensors=True
# )
# processor = AutoProcessor.from_pretrained(stt_model_id)
#
# # If stt model has not been saved, save it
# if not os.path.exists(stt_model_path):
#     stt_model.save_pretrained(stt_model_path)
#     processor.save_pretrained(stt_model_path)
#
# # Send model to gpu or cpu device
# stt_model.to(device)
#
# # Constrain the model to english language
# stt_model.generation_config.language = "<|en|>"
#
# # Create the pipeline
# stt_pipe = pipeline(
#     "automatic-speech-recognition",
#     model=stt_model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     device="cuda:1" if torch.cuda.device_count() > 1 else device
# )


# Load TTS  Model
# from TTS.config import load_config
# from TTS.api import TTS
# if not os.path.exists(tts_model_path):
#     tts_model = TTS(tts_model_id).to(device)
#     TTS().manager.create_dir_and_download_model(
#         model_name=tts_model_id,
#         output_path=tts_model_path,
#         model_item={
#             "tos_agreed":
#               "tos_agreed.txt",
#             "github_rls_url":
#               "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--tacotron2-DDC.zip"
#         }
#     )
# else:
#     config = load_config(tts_config_path)
#     tts_model = TTS()
#     tts_model.load_tts_model_by_path(
#         model_path=f"{tts_model_id}/model_file.pth",
#         config_path=tts_config_path,
#         gpu=True if device != "cpu" else False
#     )

def file_cleanup(filename):
    os.remove(filename)
