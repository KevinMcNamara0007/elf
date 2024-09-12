import glob
import os
import shutil
import subprocess
import platform
import sys
import importlib.metadata

available_os = {"Darwin": "MacOS", "Windows": "Windows", "Linux": "Linux"}
project_root = os.getcwd()
onnxruntime_genai_repo = "https://github.com/microsoft/onnxruntime-genai"
onnxruntime_dir = "onnxruntime"
onnxruntime_genai_dir = "onnxruntime-genai"
ort_dir = os.path.join(onnxruntime_genai_dir, "ort")
current_os = available_os.get(platform.system())


def check_cuda():
    try:
        result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.returncode == 0
    except FileNotFoundError:
        print("CUDA not found. CUDA support will not be enabled.")
        return False
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False


def ensure_dependencies():
    """Ensure that required dependencies are installed and up-to-date."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "wheel", "setuptools"], check=True)
        print("Dependencies updated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error updating dependencies: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error in ensure_dependencies: {e}")
        sys.exit(1)


def clone_repo(repo_url, repo_dir):
    try:
        if not os.path.exists(repo_dir):
            subprocess.run(["git", "clone", repo_url], check=True)
            print(f"Cloned repository {repo_url} into {repo_dir}")
        else:
            print(f"{repo_dir} already exists. Skipping clone")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository {repo_url}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error in clone_repo: {e}")
        sys.exit(1)


def download_onnxruntime_binaries():
    try:
        if not os.path.exists(ort_dir):
            if current_os == "Windows":
                url = "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip"
                file_name = "onnxruntime-win-x64-1.19.2.zip"
                extract_cmd = ["tar", "xvf", file_name]
                extracted_dir = "onnxruntime-win-x64-1.19.2"
            elif current_os == "Linux":
                url = "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz"
                file_name = "onnxruntime-linux-x64-gpu-1.19.2.tgz"
                extract_cmd = ["tar", "xvzf", file_name]
                extracted_dir = "onnxruntime-linux-x64-gpu-1.19.2"
            elif current_os == "MacOS":
                url = "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-universal2-1.19.2.tgz"
                file_name = "onnxruntime-osx-universal2-1.19.2.tgz"
                extract_cmd = ["tar", "xvzf", file_name]
                extracted_dir = "onnxruntime-osx-universal2-1.19.2"
            else:
                raise NotImplementedError("Unsupported OS")

            subprocess.run(["curl", "-L", url, "-o", file_name], check=True)
            subprocess.run(extract_cmd, check=True)

            if os.path.exists(ort_dir):
                shutil.rmtree(ort_dir)
            os.makedirs(ort_dir, exist_ok=True)

            for item in os.listdir(extracted_dir):
                src_path = os.path.join(extracted_dir, item)
                dst_path = os.path.join(ort_dir, item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

            shutil.rmtree(extracted_dir)
            print("ONNX Runtime binaries downloaded and extracted successfully.")
        else:
            print("ONNX Runtime binaries are already downloaded and extracted.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading or extracting ONNX Runtime: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error in download_onnxruntime_binaries: {e}")
        sys.exit(1)


def build_onnxruntime_genai(repo_dir):
    try:
        ensure_dependencies()

        wheel_files, wheel_file_dir = find_wheel_file(onnxruntime_genai_dir)

        if not wheel_files:
            os.chdir(repo_dir)
            build_command = ["python", "build.py"] if current_os == "Windows" else ["python3", "build.py"]
            if current_os == "Windows":
                build_command.extend(["--config", "Release"])
            elif current_os == "Linux":
                build_command.extend(["--config", "Release"])
                if check_cuda():
                    build_command.append("--use_cuda")
            elif current_os == "MacOS":
                build_command.extend(["--config", "Release"])
            else:
                raise NotImplementedError("Unsupported OS")

            subprocess.run(build_command, check=True)
            print("onnxruntime-genai built successfully.")
        else:
            print("Build folder present, skipping build.")
    except subprocess.CalledProcessError as e:
        print(f"Error building onnxruntime-genai: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error in build_onnxruntime_genai: {e}")
        sys.exit(1)


def install_wheel(wheel_files, wheel_file_dir):
    try:
        if not wheel_files:
            raise FileNotFoundError(f"No wheel files found in {wheel_file_dir}")
        wheel_path = os.path.join(wheel_file_dir, wheel_files[0])
        install_command = [sys.executable, "-m", "pip", "install", wheel_path]
        subprocess.run(install_command, check=True)
        print(f"Wheel file {wheel_path} installed successfully.")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error installing wheel file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error in install_wheel: {e}")
        sys.exit(1)


def check_package(package_name):
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def find_wheel_file(repo_dir="onnxruntime-genai"):
    try:
        wheel_pattern = os.path.join(project_root, repo_dir, "build", current_os, "Release", "wheel", "*.whl")
        wheel_files = glob.glob(wheel_pattern)
        wheel_file_dir = os.path.dirname(wheel_pattern)
        return wheel_files, wheel_file_dir
    except Exception as e:
        print(f"Unexpected error in find_wheel_file: {e}")
        sys.exit(1)


def main():
    try:
        ensure_dependencies()

        if not check_package("onnxruntime-genai"):
            clone_repo(onnxruntime_genai_repo, onnxruntime_genai_dir)
            download_onnxruntime_binaries()
            build_onnxruntime_genai(onnxruntime_genai_dir)
            wheel_files, wheel_file_dir = find_wheel_file(onnxruntime_genai_dir)
            install_wheel(wheel_files, wheel_file_dir)
        else:
            print("onnxruntime-genai is already installed.")
        print(os.getcwd())
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
