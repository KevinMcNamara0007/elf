import os
import shutil
import subprocess
import platform
import sys

onnxruntime_genai_repo = "https://github.com/microsoft/onnxruntime-genai"
onnxruntime_dir = "onnxruntime"
onnxruntime_genai_dir = "onnxruntime-genai"
ort_dir = os.path.join(onnxruntime_genai_dir, "ort")


def check_cuda():
    try:
        result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.returncode == 0
    except Exception as e:
        print(str(e))
        return False


def clone_repo(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print(f"{repo_dir} already exists. Skipping clone")


def download_onnxruntime_binaries():
    if platform.system() == "Windows":
        url = "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip"
        file_name = "onnxruntime-win-x64-1.19.2.zip"
        extract_cmd = ["tar", "xvf", file_name]
        extracted_dir = "onnxruntime-win-x64-1.19.2"
    elif platform.system() == "Linux":
        url = "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz"
        file_name = "onnxruntime-linux-x64-gpu-1.19.2.tgz"
        extract_cmd = ["tar", "xvzf", file_name]
        extracted_dir = "onnxruntime-linux-x64-gpu-1.19.2"
    elif platform.system() == "Darwin":  # macOS
        url = "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-universal2-1.19.2.tgz"
        file_name = "onnxruntime-osx-universal2-1.19.2.tgz"
        extract_cmd = ["tar", "xvzf", file_name]
        extracted_dir = "onnxruntime-osx-universal2-1.19.2"
    else:
        raise NotImplementedError("Unsupported OS")

    # Download ONNX Runtime binaries
    subprocess.run(["curl", "-L", url, "-o", file_name], check=True)

    # Extract ONNX Runtime binaries
    subprocess.run(extract_cmd, check=True)

    # Clean up the existing ort directory if it exists
    if os.path.exists(ort_dir):
        shutil.rmtree(ort_dir)

    # Create ort directory if it does not exist
    if not os.path.exists(ort_dir):
        os.makedirs(ort_dir)

    # Move contents from the extracted directory to ort directory
    temp_dir = extracted_dir  # Temporary directory with extracted files
    for item in os.listdir(temp_dir):
        src_path = os.path.join(temp_dir, item)
        dst_path = os.path.join(ort_dir, item)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

    # Clean up the temporary extracted directory
    shutil.rmtree(temp_dir)


def build_onnxruntime_genai(repo_dir):
    os.chdir(repo_dir)
    build_command = ["python", "build.py"] if platform.system() == "Windows" else ["python3", "build.py"]
    # Add specific flags for different platforms
    if platform.system() == "Windows":
        build_command.extend(["--config", "Release"])
    elif platform.system() == "Linux":
        build_command.extend(["--config", "Release"])
        if check_cuda():
            build_command.append("--use_cuda")
    elif platform.system() == "Darwin":  # macOS
        build_command.extend(["--config", "Release"])
    else:
        raise NotImplementedError("Unsupported OS")

    subprocess.run(build_command, check=True)


def install_wheel(repo_dir):
    wheel_dir = os.path.join("build", "Windows", "Release")
    if not os.path.exists(wheel_dir):
        raise FileNotFoundError(f"Wheel directory {wheel_dir} does not exist.")
    wheel_files = [f for f in os.listdir(wheel_dir) if f.endswith(".whl")]
    if not wheel_files:
        raise FileNotFoundError(f"No wheel files in {wheel_dir}")
    wheel_path = os.path.join(wheel_dir, wheel_files[0])
    install_command = [sys.executable, "-m", "pip", "install", wheel_path] if platform.system() == "Windows" else [
        "python3", "-m", "pip", "install", wheel_path]
    subprocess.run(install_command, check=True)


def main():
    # Clone onnxruntime-genai repo
    clone_repo(onnxruntime_genai_repo, onnxruntime_genai_dir)

    # Download and extract ONNX Runtime binaries
    download_onnxruntime_binaries()

    # Build onnxruntime-genai
    build_onnxruntime_genai(onnxruntime_genai_dir)

    # Install the wheel
    install_wheel(onnxruntime_genai_dir)


if __name__ == "__main__":
    main()
