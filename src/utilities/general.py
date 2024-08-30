import os
import subprocess
import importlib.metadata
import glob
import shutil
import time

from huggingface_hub import snapshot_download, hf_hub_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.getenv("ENV", "config/.env-dev"))
FRAMEWORKS_DIR = os.getenv("FRAMEWORKS_DIR", "efs/frameworks")
BASE_MODEL_DIR = os.getenv("BASE_MODEL_DIR", "/opt/models")
VISION_MODEL_DIR = os.getenv("VISION_MODEL_DIR")
ONNX_DIR = os.path.join(FRAMEWORKS_DIR, os.getenv("ONNX_DIR", "onnxruntime"))
ONNX_GENAI_DIR = os.path.join(FRAMEWORKS_DIR, os.getenv("ONNX_GENAI_DIR", "onnxruntime-genai"))
CONFIG_OPTION = os.getenv("CONFIG_OPTION", "RelWithDebInfo")
onnx_model_dir = os.path.join(BASE_MODEL_DIR, VISION_MODEL_DIR)


def run_subprocess(command, cwd=None):
    """Run a subprocess command with error handling and output capture."""
    try:
        result = subprocess.run(command.split(), check=True, cwd=cwd, capture_output=True, text=True, start_new_session=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")
        print(e.stderr)
        raise


def is_package_installed(package_name):
    """Check if a package is installed."""
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def file_finder(location_path):
    file_path = glob.glob(location_path, recursive=True)
    if not file_path:
        print(f"No file found in the path: {location_path}")
    return file_path


def download_vision_model():
    """Downloads the Vision model and saves it in VISION_MODEL_DIR."""
    repo_id = "microsoft/Phi-3-vision-128k-instruct-onnx-cpu"
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


def clone_repo(url, local_dir, tries=3):
    try:
        run_subprocess(f"git clone {url} {local_dir}")
    except Exception as e:
        print(f"Error cloning repository {url}: {e}")
        if tries > 0:
            shutil.rmtree(local_dir, ignore_errors=True)
            clone_repo(url, local_dir, tries=tries - 1)
        raise Exception(f"Error cloning repository.")


def run_command_with_retry(command, cwd=None, tries=3):
    try:
        run_subprocess(command)
    except Exception as e:
        print(f"Error building source: {e}")
        if tries > 0:
            run_command_with_retry(command, cwd, tries=tries - 1)


def copy_files(dir_paths, files_to_copy):
    # Copy build files to onnxruntime-genai directory
    for dir_path, file_path in zip(dir_paths, files_to_copy):
        os.makedirs(dir_path, exist_ok=True)
        shutil.copy(file_path, dir_path)


def build_onnx_runtime_genai():
    """Builds and installs onnxruntime-genai from source."""
    # Save the original working directory
    original_dir = os.getcwd()

    try:
        # Check if onnxruntime-genai is already installed
        if is_package_installed("onnxruntime-genai"):
            print("\nonnxruntime-genai is already installed. Skipping build process.\n")
            return

        # Ensure ONNX directories exist or clone the repos
        if not os.path.exists(ONNX_GENAI_DIR):
            print(f"\nCloning onnxruntime-genai into {ONNX_GENAI_DIR}...\n")
            clone_repo("https://github.com/microsoft/onnxruntime-genai.git", ONNX_GENAI_DIR)

        if not os.path.exists(ONNX_DIR):
            print(f"\nCloning onnxruntime into {ONNX_DIR}...\n")
            clone_repo("https://github.com/microsoft/onnxruntime.git", ONNX_DIR)

        # Check for build files
        print("\nBuild file check...\n")
        dylib_files = file_finder(f"build/MacOS/{CONFIG_OPTION}/libonnxruntime*.dylib*")
        if not dylib_files:
            os.chdir(ONNX_DIR)
            print("\nGenerating build files...\n")
            run_command_with_retry(
                command=f"python3 build.py --build_shared_lib --skip_tests --parallel --config {CONFIG_OPTION} --allow_running_as_root",
            )
        dylib_files = file_finder(f"build/MacOS/{CONFIG_OPTION}/libonnxruntime*.dylib*")
        files_to_copy = ["include/onnxruntime/core/session/onnxruntime_c_api.h"]
        files_to_copy.extend(dylib_files)
        dir_paths = ["../onnxruntime-genai/ort/include"]
        dir_paths.extend(["../onnxruntime-genai/ort/lib" for _ in range(len(dylib_files))])
        copy_files(dir_paths, files_to_copy)

        # Check for wheel
        os.chdir("../onnxruntime-genai")
        og_wheels = glob.glob(f"build/macOS/{CONFIG_OPTION}/wheel/*.whl")
        if not og_wheels:
            # Build onnxruntime-genai
            print("\nBuilding wheel...\n")
            os.chdir("../onnxruntime-genai")
            run_command_with_retry(
                command="python3 build.py"
            )
        wheel_path = glob.glob(f"../onnxruntime-genai/build/macOS/{CONFIG_OPTION}/wheel/*.whl")
        print(f"\nInstalling onnxruntime-genai wheel file: {wheel_path[0]}\n")
        run_command_with_retry(f"python3 -m pip install {wheel_path[0]}")
        if not is_package_installed("onnxruntime-genai"):
            print("\nPackage installation failed. Retrying...\n")
            time.sleep(5)
            run_command_with_retry(f"python3 -m pip install {wheel_path[0]}", tries=5)
        print("Installed onnxruntime-genai")
    finally:
        # Change back to the original working directory
        os.chdir(original_dir)
        print("Returned to the original directory")
