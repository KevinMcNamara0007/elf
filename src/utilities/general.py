import os
import pickle
import subprocess
from dotenv import load_dotenv
from llama_cpp import Llama
from tensorflow.keras.models import load_model


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
context_window = 8196

# Available Expert Models
general_expert = Llama(
    model_path=general_model_path,
    n_gpu_layers=-1,
    n_ctx=context_window,
    top_p=0.6,
    top_k=10,
    use_gpu=True
)
# programming_expert = Llama(model_path=programming_model_path, n_gpu_layers=-1, n_ctx=2048)

# Classifications
classifications = {
    0: {"Model": general_expert, "Category": "code"},
    1: {"Model": general_expert, "Category": "language"},
    2: {"Model": general_expert, "Category": "math"}
}
