import multiprocessing
import os
import pickle
import platform
import subprocess
import shutil
import torch
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Global LLAMA_SERVER_PID variable
LLAMA_SERVER_PID = None

# Import ENV Vars
load_dotenv(os.getenv("ENV", "config/.env-dev"))
general_model_path = os.getenv("general")
programming_model_path = os.getenv("programming")
classifier_encoder = os.getenv("classifier_encoder")
classifier_tokenizer = os.getenv("classifier_tokenizer")
classifier_model = os.getenv("classifier_model")
stt_model_path = os.getenv("stt_model_path")
tts_model_path = os.getenv("tts_model_path")
tts_config_path = os.getenv("tts_config_path")
vision_model_path = os.getenv("vision_model_path")
CONTEXT_WINDOW = os.getenv("CONTEXT_WINDOW")
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
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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


# Ensure llama.cpp repository exists
ensure_llama_cpp_repository()


def compile_llama_cpp():
    # Ensure LLAMA_CPP_HOME directory exists
    if not os.path.exists(LLAMA_CPP_HOME):
        # Change directory to LLAMA_CPP_HOME
        os.makedirs(LLAMA_CPP_HOME)
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
                ["cmake", "--build", ".", "--config", "Release", "--target", "llama-server", WORKERS],
                check=True,
                capture_output=True,
            )
            print(build_command.stdout.decode())
            print(build_command.stderr.decode())

        except subprocess.CalledProcessError as e:
            print(f"Error during cmake or build process: {e}")
            raise e


compile_llama_cpp()


def start_llama_cpp():
    global LLAMA_SERVER_PID

    # Change working directory to LLAMA_CPP_HOME for starting llama-server
    cwd = os.getcwd()
    os.chdir(LLAMA_CPP_HOME)

    llama_cpp_process = subprocess.Popen(
        [LLAMA_CPP_PATH, "--model", GENERAL_MODEL_PATH, "--ctx-size", CONTEXT_WINDOW, "--port", str(LLAMA_PORT), "-np",
         "2", "-ns", "1", "-ngl", "16", "-sm", "layer", "-ts", "0", "-mg", "-1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    LLAMA_SERVER_PID = llama_cpp_process.pid

    # Change back to the original working directory
    os.chdir(cwd)


def stop_llama_cpp():
    global LLAMA_SERVER_PID
    if LLAMA_SERVER_PID:
        os.kill(int(LLAMA_SERVER_PID), 9)  # Terminate the process
        LLAMA_SERVER_PID = None


# Capture SIGTERM signal to stop llama-server
def handle_sigterm(signum, frame):
    stop_llama_cpp()
    exit(0)


def replace_port_in_url(url, new_port):
    parts = url.split(":")
    parts[-1] = str(new_port) + "/" + parts[-1].split("/")[-1]  # Replace the second last part (the port)
    return ":".join(parts)


# Load the tokenizer (assuming it has been saved as instructed)
with open(classifier_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load Classifier Model
classifier = load_model(classifier_model)

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
