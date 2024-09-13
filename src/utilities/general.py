import os
from huggingface_hub import snapshot_download, hf_hub_download
from dotenv import load_dotenv

from src.utilities.genai_builder import check_gpu_linux

# Load environment variables
load_dotenv(os.getenv("ENV", "config/.env-dev"))
BASE_MODEL_DIR = os.getenv("BASE_MODEL_DIR", "/opt/models")
VISION_MODEL_DIR = os.getenv("VISION_MODEL_DIR")
onnx_model_dir = os.path.join(BASE_MODEL_DIR, VISION_MODEL_DIR)


def download_vision_model():
    """Downloads the Vision model and saves it in VISION_MODEL_DIR."""
    repo_id = "microsoft/Phi-3-vision-128k-instruct-onnx-cpu" if not check_gpu_linux() else "microsoft/Phi-3-vision-128k-instruct-onnx-cuda"
    expected_files = [
        "genai_config.json",
        "phi-3-v-128k-instruct-text-embedding.onnx",
        "phi-3-v-128k-instruct-text-embedding.onnx.data",
        "phi-3-v-128k-instruct-text.onnx",
        "phi-3-v-128k-instruct-text.onnx.data",
        "phi-3-v-128k-instruct-vision.onnx",
        "phi-3-v-128k-instruct-vision.onnx.data",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    # Download the repository
    partial_dir = '/'.join(onnx_model_dir.split("/")[:-1])
    if not os.path.exists(onnx_model_dir):
        os.makedirs(partial_dir, exist_ok=True)
        print(f"Downloading the repository {VISION_MODEL_DIR} to {partial_dir}...")
        try:
            snapshot_download(repo_id=repo_id, local_dir=partial_dir)
        except Exception as e:
            print(f"Error downloading repository Phi-3-vision: {e}")
            return

    # Check for missing files
    downloaded_files = os.listdir(onnx_model_dir)
    missing_files = [f for f in expected_files if f not in downloaded_files]

    if missing_files:
        print(f"Missing files: {missing_files}. Re-downloading missing files...")
        for file in missing_files:
            try:
                # Download missing files
                hf_hub_download(repo_id=repo_id, filename=file, local_dir=partial_dir)
            except Exception as e:
                print(f"Error downloading missing file {file}: {e}")
