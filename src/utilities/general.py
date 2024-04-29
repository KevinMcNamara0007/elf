import os
import pickle
import subprocess
import warnings
import torch
from dotenv import load_dotenv
from llama_cpp import Llama
from parler_tts import ParlerTTSForConditionalGeneration
from tensorflow.keras.models import load_model
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, VitsModel, AutoTokenizer

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

# Llama cpp install
os.environ["CMAKE_ARGS"] = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
subprocess.run(["pip", "install", "llama-cpp-python"])

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

stt_model_id = stt_model_path if os.path.exists(stt_model_path) else "openai/whisper-medium"
tts_model_id = tts_model_path if os.path.exists(tts_model_path) else "parler-tts/parler_tts_mini_v0.1"

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    stt_model_id,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
# If stt model has not been saved, save it
if not os.path.exists(stt_model_path):
    stt_model.save_pretrained(stt_model_path)

# Send model to gpu or cpu device
stt_model.to(device)

# Constrain the model to english language
stt_model.generation_config.language = "<|en|>"

# Create the pipeline
processor = AutoProcessor.from_pretrained(stt_model_id)
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

# Load the model
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_id).to(device)
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_id)


# If tts model has not been saved, save it
if not os.path.exists(tts_model_path):
    tts_model.save_pretrained(tts_model_path)
    tts_tokenizer.save_pretrained(tts_model_path)


def file_cleanup(filename):
    os.remove(filename)
