import importlib.metadata
import os
import subprocess
import shutil
import platform
import urllib.request
import zipfile
import tarfile

# Define constants
PLATFORM = platform.system() if platform.system() != 'Darwin' else "MacOS"
REPO_URL = "https://github.com/microsoft/onnxruntime-genai.git"
REPO_DIR = "onnxruntime-genai"
ORT_REPO_URL = "https://github.com/Microsoft/onnxruntime.git"
ORT_REPO_DIR = "onnxruntime"
ORT_DIR = os.path.join(REPO_DIR, "ort")
SETUP_PY_DIR = os.path.join(REPO_DIR, "build", PLATFORM, "Release", "wheel")
DIST_DIR = os.path.join(SETUP_PY_DIR, "dist")
BUILD_DIR = os.path.join(ORT_REPO_DIR, "build", PLATFORM, "Release", "onnxruntime")


# Supported packages of onnxruntime-genai
ONNX_RUNTIME_VERSIONS = {
    "Windows": {
        "cpu": {
            "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip",
            "filename": "onnxruntime-win-x64-1.19.2.zip",
            "extract": lambda file: _extract_zip(file, ORT_DIR),
        },
        "gpu": {
            "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-gpu-1.19.2.zip",
            "filename": "onnxruntime-win-x64-gpu-1.19.2.zip",
            "extract": lambda file: _extract_zip(file, ORT_DIR),
        }
    },
    "Linux": {
        "cpu": {
            "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz",
            "filename": "onnxruntime-linux-x64-1.19.2.tgz",
            "extract": lambda file: _extract_tar(file, ORT_DIR),
        },
        "gpu": {
            "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz",
            "filename": "onnxruntime-linux-x64-gpu-1.19.2.tgz",
            "extract": lambda file: _extract_tar(file, ORT_DIR),
        }
    },
    "MacOS": {
        "cpu": {
            "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-osx-universal2-1.19.2.tgz",
            "filename": "onnxruntime-osx-universal2-1.19.2.tgz",
            "extract": lambda file: _extract_tar(file, ORT_DIR),
        }
    }
}


def _extract_zip(file, target_dir):
    """
    Extract zip file to target_dir.
    :param file:
    :param target_dir:
    :return:
    """
    with zipfile.ZipFile(file, 'r') as z:
        for member in z.namelist():
            if 'include' in member:
                member_path = os.path.join(target_dir, 'include',
                                           os.path.relpath(member, 'onnxruntime-win-x64-1.19.2/include'))
            elif 'lib' in member:
                member_path = os.path.join(target_dir, 'lib', os.path.relpath(member, 'onnxruntime-win-x64-1.19.2/lib'))
            else:
                continue

            if member.endswith('/'):
                os.makedirs(member_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(member_path), exist_ok=True)
                with z.open(member) as source, open(member_path, 'wb') as target:
                    shutil.copyfileobj(source, target)


def _extract_tar(file, target_dir):
    """
    Extract tar file to target_dir.
    :param file:
    :param target_dir:
    :return:
    """
    with tarfile.open(file, 'r:*') as tar:
        for member in tar.getmembers():
            if 'include' in member.name:
                member_path = os.path.join(target_dir, 'include',
                                           os.path.relpath(member.name,
                                                           'onnxruntime-linux-x64-1.19.2/include' if PLATFORM != 'MacOS' else 'onnxruntime-osx-universal2-1.19.2/include'))
            elif 'lib' in member.name:
                member_path = os.path.join(target_dir, 'lib',
                                           os.path.relpath(member.name,
                                                           'onnxruntime-linux-x64-1.19.2/lib' if PLATFORM != 'MacOS' else 'onnxruntime-osx-universal2-1.19.2/lib'))
            else:
                continue

            if member.isdir():
                os.makedirs(member_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(member_path), exist_ok=True)
                with tar.extractfile(member) as source, open(member_path, 'wb') as target:
                    shutil.copyfileobj(source, target)


def run_command(command, cwd=None, check=True):
    """
    Run command and return output.
    :param command:
    :param cwd:
    :param check:
    :return:
    """
    try:
        print(f"Running command in directory {cwd}: {' '.join(command)}")
        subprocess.run(command, cwd=cwd, check=check)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        raise


def download_and_extract(url, filename, extract_func):
    """
    Download and extract file from url.
    :param url:
    :param filename:
    :param extract_func:
    :return:
    """
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Extracting {filename}...")
        extract_func(filename)
    except Exception as e:
        print(f"Failed to download or extract {filename}: {e}")
        raise


def clone_repo(repo_url=REPO_URL, repo_dir=REPO_DIR):
    """
    Clone repository.
    :return:
    """
    if not os.path.exists(repo_dir):
        run_command(["git", "clone", "--recursive", repo_url, repo_dir])
    else:
        print(f"Repository already exists at {repo_dir}")


def check_gpu_windows():
    """
    Check if windows host has gpu support.
    :return:
    """
    try:
        # This command checks if DirectML is available, indicating GPU support.
        result = subprocess.run(["python", "-c", "import win32api; print(win32api.GetVersion())"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        print("DirectML not found. GPU support may not be enabled.")
        return False
    except Exception as e:
        print(f"Error checking GPU on Windows: {e}")
        return False


def check_gpu_linux():
    """
    Check if linux host has gpu support.
    :return:
    """
    try:
        # This command checks for CUDA installation, indicating GPU support.
        result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        print("CUDA not found. GPU support may not be enabled.")
        return False
    except Exception as e:
        print(f"Error checking GPU on Linux: {e}")
        return False


def download_onnxruntime_binaries():
    """
    Download ONNX runtime binaries. For file extraction into onnxruntime-genai/ort folder.
    :return:
    """
    current_os = PLATFORM
    if current_os not in ONNX_RUNTIME_VERSIONS:
        raise NotImplementedError(f"Unsupported OS: {current_os}")

    ort_dir_exists = os.path.exists(ORT_DIR)
    if not ort_dir_exists:
        os.makedirs(ORT_DIR, exist_ok=True)

    gpu_supported = False
    if current_os == "Windows":
        gpu_supported = check_gpu_windows()
    elif current_os == "Linux":
        gpu_supported = check_gpu_linux()

    if gpu_supported:
        version = "gpu"
    else:
        version = "cpu"

    config = ONNX_RUNTIME_VERSIONS[current_os].get(version, ONNX_RUNTIME_VERSIONS[current_os]['cpu'])
    download_and_extract(config["url"], config["filename"], config["extract"])


def build_generate_api():
    """
    Builds the onnxruntime-genai package
    :return:
    """
    try:
        print("Building generate() API...")
        build_command = ["python" if platform.system() == "Windows" else "python3", "build.py", "--config", "Release"]
        if platform.system() == "Linux" and check_gpu_linux():
            build_command += ["--use_cuda"]
        elif platform.system() == "Windows" and check_gpu_windows():
            build_command += ["--use_dml"]

        print(f"Running build command: {' '.join(build_command)}")
        run_command(build_command, cwd=REPO_DIR)

    except Exception as e:
        if os.path.exists(os.path.join(REPO_DIR, "build", PLATFORM, "Release", "wheel", "setup.py")):
            build_wheel()
        else:
            print(f"Failed to build wheel: {str(e)}")
            raise


def build_wheel():
    """
    Builds the onnxruntime-genai's wheel for pip install.
    :return:
    """
    try:
        print("Building wheel from setup.py...")
        build_command = ["python" if PLATFORM == "Windows" else "python3", "setup.py", "bdist_wheel"]
        print(f"Running build command: {' '.join(build_command)}")
        run_command(build_command, cwd=SETUP_PY_DIR)

    except Exception as e:
        print(f"Build failed: {e}")
        raise


def install_library():
    """
    Installs the onnxruntime-genai wheel into pip's site-packages.
    :return:
    """
    try:
        print("Installing library...")

        # Ensure the distribution directory exists
        if not os.path.exists(DIST_DIR) and PLATFORM == "Windows":
            raise FileNotFoundError(
                f"{DIST_DIR} directory does not exist. Ensure that the wheel was built successfully."
            )

        # Find the wheel file in the correct directory based on OS
        wheel_dir = DIST_DIR if PLATFORM == "Windows" else SETUP_PY_DIR
        wheel_files = [f for f in os.listdir(wheel_dir) if f.endswith('.whl')]

        if not wheel_files:
            raise FileNotFoundError("No wheel files found in the specified directory.")

        # Install the found wheel file
        if PLATFORM == "Windows":
            run_command(["pip", "install", wheel_files[0]], cwd=DIST_DIR)
        elif PLATFORM in ["Linux", "MacOS"]:  # Darwin is for MacOS
            run_command(["pip3", "install", wheel_files[0]], cwd=SETUP_PY_DIR)
        else:
            raise NotImplementedError(f"Unsupported OS: {PLATFORM}")
    except Exception as e:
        print(f"Installation failed: {e}")
        raise


def check_package(package_name):
    """
    Checks if package is installed in pip.
    :param package_name:
    :return:
    """
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def set_environment_variables():
    """
    Set necessary environment variables for locating shared libraries.
    :return:
    """
    ort_lib_dir = os.path.join(ORT_DIR, "lib")
    if PLATFORM == "MacOS":
        os.environ["DYLD_LIBRARY_PATH"] = f"{ort_lib_dir}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
    elif PLATFORM == "Linux":
        os.environ["LD_LIBRARY_PATH"] = f"{ort_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    elif PLATFORM == "Windows":
        os.environ["PATH"] = f"{ort_lib_dir}:{os.environ.get('PATH', '')}"
    print(f"Environment variables set for {PLATFORM}.")


def build_onnxruntime():
    """
    Clone and build onnxruntime from source
    :return:
    """
    if not os.path.exists(ORT_REPO_DIR):
        run_command(["git", "clone", ORT_REPO_URL, ORT_REPO_DIR])

    if PLATFORM == "Windows":
        run_command(["python", "-m", "pip", "install", "cmake"])
        run_command(["which", "cmake"])
        run_command([".\\build.bat", "--config", "RelWithDebInfo", "--build_shared_lib", "--parallel", "--compile_no_warning_as_error", "--skip-tests"], cwd=ORT_REPO_DIR)
    if PLATFORM in ["Linux", "MacOS"]:
        run_command(["./build.sh", "--config", "RelWithDebInfo", "--build_shared_lib", "--parallel", "--compile_no_warning_as_error", "--skip_tests"], cwd=ORT_REPO_DIR)


def copy_built_files():
    """
    Copy built onnxruntime files from the build directory to the ort directory.
    :return:
    """
    if not os.path.exists(BUILD_DIR):
        raise FileNotFoundError(f"{BUILD_DIR} does not exist. Ensure that onnxruntime is built successfully.")

    os.makedirs(ORT_DIR, exist_ok=True)

    # Copy include and lib directories
    for folder in ["include", "lib"]:
        src_dir = os.path.join(BUILD_DIR, folder)
        dest_dir = os.path.join(ORT_DIR, folder)
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        else:
            print(f"{src_dir} does not exist. Skipping...")


def build_onnxruntime_genai():
    if not check_package("onnxruntime-genai"):
        set_environment_variables()
        build_onnxruntime()
        copy_built_files()
        if not os.path.exists(REPO_DIR):
            clone_repo()
        wheel_files = [f for f in os.listdir(DIST_DIR if PLATFORM == "Windows" else SETUP_PY_DIR) if f.endswith('.whl')]
        if not wheel_files:
            build_generate_api()
        install_library()
    else:
        print("onnxruntime-genai package is already installed.")