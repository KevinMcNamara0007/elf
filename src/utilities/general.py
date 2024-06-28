import os
import pickle
import subprocess
import warnings
import torch
from dotenv import load_dotenv
from llama_cpp import Llama
from tensorflow.keras.models import load_model
from transformers import (
    AutoModelForSpeechSeq2Seq,
    pipeline,
    AutoModelForCausalLM,
    AutoProcessor
)

warnings.filterwarnings('ignore')

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

# Llama cpp install
os.environ["CMAKE_ARGS"] = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
subprocess.run(["pip", "install", "llama-cpp-python"])
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

stt_model_id = stt_model_path if os.path.exists(stt_model_path) else "openai/whisper-medium"
tts_model_id = tts_model_path if os.path.exists(tts_model_path) else "tts_models/en/ljspeech/tacotron2-DDC"

# Load the tokenizer (assuming it has been saved as instructed)
with open(classifier_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

classifier = load_model(classifier_model)
context_window = 8196

# Available Expert Models
general_expert = Llama(
    model_path=general_model_path,
    n_ctx=context_window,
    top_p=0.6,
    top_k=10,
    use_gpu=True
)
# programming_expert = Llama(model_path=programming_model_path, n_gpu_layers=-1, n_ctx=2048)

# Load vision model
vision_model_id = vision_model_path if os.path.exists(vision_model_path) else "microsoft/Phi-3-vision-128k-instruct"

vision_model = AutoModelForCausalLM.from_pretrained(
    vision_model_id,
    device_map=device,
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager"  # use _attn_implementation='flash_attention_2' to enable flash attention
)

vision_processor = AutoProcessor.from_pretrained(
    vision_model_id,
    trust_remote_code=True
)

if not os.path.exists(vision_model_path):
    vision_model.save_pretrained(
        vision_model_path,
        is_main_process=True,
        save_functional=True,
        save_classif_head=True,
        save_tokenizer=True,
        save_shared=True,  # Ensure shared tensors are saved
        safe_serialization=False  # Bypass safety check for shared tensors
    )
    vision_processor.save_pretrained(
        vision_model_path,
        is_main_process=True,
        save_functional=True,
        save_classif_head=True,
        save_tokenizer=True,
        save_shared=True,  # Ensure shared tensors are saved
        safe_serialization=False  # Bypass safety check for shared tensors
    )

# Classifications
classifications = {
    0: {"Model": general_expert, "Category": "code"},
    1: {"Model": general_expert, "Category": "language"},
    2: {"Model": general_expert, "Category": "math"}
}

# Load Speech to Text Model
stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    stt_model_id,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
processor = AutoProcessor.from_pretrained(stt_model_id)

# If stt model has not been saved, save it
if not os.path.exists(stt_model_path):
    stt_model.save_pretrained(stt_model_path)
    processor.save_pretrained(stt_model_path)

# Send model to gpu or cpu device
stt_model.to(device)

# Constrain the model to english language
stt_model.generation_config.language = "<|en|>"

# Create the pipeline
stt_pipe = pipeline(
    "automatic-speech-recognition",
    model=stt_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    device="cuda:1" if torch.cuda.device_count() > 1 else device
)


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
