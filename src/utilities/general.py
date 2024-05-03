import os
import pickle
import subprocess
import warnings
import torch
from TTS.api import TTS
from dotenv import load_dotenv
from llama_cpp import Llama
from tensorflow.keras.models import load_model
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sys

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# Import ENV Vars
load_dotenv(os.getenv("ENV", "src/config/.env-dev"))
general_model_path = os.getenv("general")
programming_model_path = os.getenv("programming")
classifier_encoder = os.getenv("classifier_encoder")
classifier_tokenizer = os.getenv("classifier_tokenizer")
classifier_model = os.getenv("classifier_model")
stt_model_path = os.getenv("stt_model_path")
tts_model_path = os.getenv("tts_model_path")
tts_config_path = os.getenv("tts_config_path")
tts_vocoder_path = os.getenv("tts_vocoder_path")
tts_vocoder_config_path = os.getenv("tts_vocoder_config_path")

# Llama cpp install
os.environ["CMAKE_ARGS"] = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
subprocess.run(["pip", "install", "llama-cpp-python"])
subprocess.run(["pip", "install", "numpy", "--upgrade"])

stt_model_id = stt_model_path if os.path.exists(stt_model_path) else "openai/whisper-medium"
tts_model_id = tts_model_path if os.path.exists(tts_model_path) else "tts_models/en/ljspeech/glow-tts"
tts_vocoder_id = tts_vocoder_path if os.path.exists(tts_vocoder_path) else "vocoder_models/en/ljspeech/hifigan_v2"

# Load the tokenizer (assuming it has been saved as instructed)
with open(classifier_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

classifier = load_model(classifier_model)

# Available Expert Models
general_expert = Llama(model_path=general_model_path, n_gpu_layers=-1, n_ctx=8196)

# Classifications
classifications = {
    0: {"Model": general_expert, "Category": "code"},
    1: {"Model": general_expert, "Category": "language"},
    2: {"Model": general_expert, "Category": "math"}
}

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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
    device=device
)


if os.path.exists(tts_model_path):
    tts_model = TTS()
    tts_model.load_tts_model_by_path(
        model_path=f"{tts_model_path}/model_file.pth",
        config_path=tts_config_path,
        vocoder_path=f"{tts_vocoder_path}/model_file.pth",
        vocoder_config=tts_vocoder_config_path,
        gpu=True if device != "cpu" else False
    )
else:
    tts_model = TTS(tts_model_id).to(device)
    TTS().manager.create_dir_and_download_model(
        model_name=tts_model_id,
        model_item={
            "tos_agreed": "tos_agreed.txt",
            "license": "Apache-2.0",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--glow-tts.zip"
        },
        output_path=tts_model_path
    )
    TTS().manager.create_dir_and_download_model(
        model_name=tts_vocoder_id,
        model_item={
            "tos_agreed": "tos_agreed.txt",
            "license": "Apache-2.0",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ljspeech--multiband"
                              "-melgan.zip"
        },
        output_path=tts_vocoder_path
    )


def file_cleanup(filename):
    os.remove(filename)
