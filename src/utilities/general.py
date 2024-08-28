import os
import subprocess
import onnxruntime_genai as og
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.getenv("ENV", "config/.env-dev"))
FRAMEWORKS_DIR = os.getenv("FRAMEWORKS_DIR")
BASE_MODEL_DIR = os.getenv("BASE_MODEL_DIR")
VISION_MODEL_DIR = os.getenv("VISION_MODEL_DIR")
ONNX_DIR = os.path.join(FRAMEWORKS_DIR, os.getenv("ONNX_DIR", ""))
ONNX_GENAI_DIR = os.path.join(FRAMEWORKS_DIR, os.getenv("ONNX_GENAI_DIR", ""))
CONFIG_OPTION = os.getenv("CONFIG_OPTION", "RelWithDebInfo")
onnx_model_dir = os.path.join(BASE_MODEL_DIR, VISION_MODEL_DIR)


def download_vision_model():
    """Downloads the Vision model and saves it in VISION_MODEL_DIR."""
    if not os.path.exists(onnx_model_dir):
        os.makedirs(os.path.dirname(onnx_model_dir), exist_ok=True)
        print(f"Downloading the repository {VISION_MODEL_DIR} to {onnx_model_dir}...")
        try:
            snapshot_download(repo_id="microsoft/Phi-3-vision-128k-instruct-onnx-cpu", local_dir=onnx_model_dir)
        except Exception as e:
            print(f"Error downloading repository Phi-3-vision: {e}")


def build_onnx_runtime_genai():
    """Builds and installs onnxruntime-genai from source."""
    if not os.path.exists(ONNX_GENAI_DIR):
        os.makedirs(FRAMEWORKS_DIR, exist_ok=True)
        root = os.getcwd()
        try:
            # Clone necessary repositories
            subprocess.run(f"git clone https://github.com/microsoft/onnxruntime-genai {FRAMEWORKS_DIR}".split(),
                           check=True)
            subprocess.run(f"git clone https://github.com/microsoft/onnxruntime.git {FRAMEWORKS_DIR}".split(),
                           check=True)

            # Build onnxruntime
            os.chdir(ONNX_DIR)
            subprocess.run(f"./build.sh --build_shared_lib --skip_tests --parallel --config {CONFIG_OPTION}".split(),
                           check=True)

            # Copy build files
            os.makedirs("../onnxruntime-genai/ort/include", exist_ok=True)
            os.makedirs("../onnxruntime-genai/ort/lib", exist_ok=True)
            subprocess.run(
                "cp include/onnxruntime/core/session/onnxruntime_c_api.h ../onnxruntime-genai/ort/include".split(),
                check=True)
            subprocess.run(
                f"cp build/MacOS/{CONFIG_OPTION}/libonnxruntime*.dylib* ../onnxruntime-genai/ort/lib".split(),
                check=True)

            # Build onnxruntime-genai
            os.chdir("../onnxruntime-genai")
            subprocess.run(f"python3 build.py --config {CONFIG_OPTION}".split(), check=True)
            os.chdir(f"build/macOS/{CONFIG_OPTION}/wheel")
            subprocess.run("python3 -m pip install *.whl".split(), check=True)
        finally:
            os.chdir(root)


# Initialize model, processor, and tokenizer
vision_model = og.Model(onnx_model_dir)
vision_processor = vision_model.create_multimodal_processor()
vision_tokenizer = vision_processor.create_stream()
