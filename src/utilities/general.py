import os
import torch
import pickle
import subprocess
from dotenv import load_dotenv
from llama_cpp import Llama
from tensorflow.keras.models import load_model
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Import ENV Vars
load_dotenv(os.getenv("ENV", "src/config/.env-dev"))
general_model_path = os.getenv("general")
programming_model_path = os.getenv("programming")
classifier_encoder = os.getenv("classifier_encoder")
classifier_tokenizer = os.getenv("classifier_tokenizer")
classifier_model = os.getenv("classifier_model")

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

if os.path.exists("efs/models/whisper-medium/distil-whisper"):
    model_id = "efs/models/whisper-medium"
    # dataset = load_from_disk("efs/models/whisper-medium/distil-whisper/librispeech_long")
else:
    model_id = "openai/whisper-medium"
    # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)
model.generation_config.language = "<|en|>"

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device
)
