import os
import subprocess

from dotenv import load_dotenv
from llama_cpp import Llama

# Import ENV Vars
load_dotenv(os.getenv("ENV", "src/config/.env-dev"))
general_model_path = os.getenv("general")
programming_model_path = os.getenv("programming")

# Llama cpp install
os.environ["CMAKE_ARGS"] = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
subprocess.run(["pip", "install", "llama-cpp-python"])

# Available Expert Models
# general_expert = Llama(model_path=general_model_path, n_gpu_layers=-1, n_ctx=8196)
# programming_expert = Llama(model_path=programming_model_path, n_gpu_layers=-1, n_ctx=2048)

# Preloaded models with categorical key
expert_models = {
    # "General": general_expert,
    # "Programming": programming_expert,
}
