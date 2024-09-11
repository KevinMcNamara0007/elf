import os
import shutil
import subprocess
import platform
import sys

onnxruntime_genai_repo = "https://github.com/microsoft/onnxruntime-genai"
onnxruntime_repo = "https://github.com/microsoft/onnxruntime"
onnxruntime_genai_dir = "onnxruntime-genai"
onnxruntime_dir = "onnxruntime"
ort_dir = os.path.join(onnxruntime_genai_dir, "ort")


def clone_repo(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print(f"{repo_dir} already exists. Skipping clone")


def check_cuda():
    try:
        result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.returncode == 0
    except Exception as e:
        print(str(e))
        return False


def get_cuda_paths():
    cuda_home = os.environ.get("CUDA_HOME", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4")
    cudnn_home = os.environ.get("CUDNN_HOME", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4")
    if not os.path.exists(cuda_home):
        raise FileNotFoundError(f"CUDA home directory {cuda_home} does not exist")
    if not os.path.exists(cudnn_home):
        raise FileNotFoundError(f"cuDNN home directory {cudnn_home} does not exist")
    return cuda_home, cudnn_home



def build_onnxruntime_windows(repo_dir, use_cuda=False):
    os.chdir(repo_dir)
    build_command = ["build.bat", "--build_shared_lib", "--skip_tests", "--parallel", "--config", "Release"]
    if use_cuda:
        cuda_home, cudnn_home = get_cuda_paths()
        print(f"Using CUDA home: {cuda_home}")
        print(f"Using cuDNN home: {cudnn_home}")
        build_command.extend(["--use_cuda", f"--cuda_home={cuda_home}", f"--cudnn_home={cudnn_home}"])
    subprocess.run(build_command, check=True)


def build_onnxruntime_unix(repo_dir, use_cuda=False):
    os.chdir(repo_dir)
    build_command = ["build.sh", "--build_shared_lib", "--skip_tests", "--parallel", "--config", "Release"]
    if use_cuda:
        cuda_home, cudnn_home = get_cuda_paths()
        build_command.extend(["--use_cuda", f"--cuda_home={cuda_home}", f"--cudnn_home={cudnn_home}"])
    subprocess.run(build_command, check=True)


def copy_built_onnxruntime_windows(src_dir, dst_dir):
    os.chdir("..")
    include_src = os.path.join(src_dir, "include", "onnxruntime", "core", "session")
    lib_src = os.path.join(src_dir, "build", "Windows", "Release")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    include_dst = os.path.join(dst_dir, "include")
    lib_dst = os.path.join(dst_dir, "lib")
    if not os.path.exists(include_dst):
        os.makedirs(include_dst)
    if not os.path.exists(lib_dst):
        os.makedirs(lib_dst)
    shutil.copytree(include_src, os.path.join(include_dst, "onnxruntime"), dirs_exist_ok=True)
    for file in os.listdir(lib_src):
        if file.startswith("onnxruntime") and file.endswith(".dll"):
            shutil.copy2(os.path.join(lib_src, file), lib_dst)
        elif file.startswith("onnxruntime") and file.endswith(".lib"):
            shutil.copy2(os.path.join(lib_src, file), lib_dst)


def copy_built_onnxruntime_unix(src_dir, dst_dir):
    os.chdir("..")
    include_src = os.path.join(src_dir, "include", "onnxruntime", "core", "session")
    lib_src = os.path.join(src_dir, "build", "Linux", "Release") if platform.system() == "Linux" else os.path.join(src_dir, "build", "MacOS", "Release")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    include_dst = os.path.join(dst_dir, "include")
    lib_dst = os.path.join(dst_dir, "lib")
    if not os.path.exists(include_dst):
        os.makedirs(include_dst)
    if not os.path.exists(lib_dst):
        os.makedirs(lib_dst)
    shutil.copytree(include_src, os.path.join(include_dst, "onnxruntime"), dirs_exist_ok=True)
    for file in os.listdir(lib_src):
        if file.startswith("libonnxruntime") and file.endswith(".so") or file.endswith(".dylib"):
            shutil.copy2(os.path.join(lib_src, file), lib_dst)


def build_onnxruntime_genai(repo_dir, ort_home=None):
    os.chdir(repo_dir)
    build_command = ["python", "build.py"] if platform.system() == "Windows" else ["python3", "build.py"]
    subprocess.run(build_command, check=True)

def install_wheel(repo_dir):
    wheel_dir = os.path.join(repo_dir, "build", "wheel")
    if not os.path.exists(wheel_dir):
        raise FileNotFoundError(f"Wheel directory {wheel_dir} does not exist.")
    wheel_files = [f for f in os.listdir(wheel_dir) if f.endswith(".whl")]
    if not wheel_files:
        raise FileNotFoundError(f"No wheel files in {wheel_dir}")
    wheel_path = os.path.join(wheel_dir, wheel_files[0])
    install_command = [sys.executable,"-m", "pip", "install", wheel_path] if platform.system() == "Windows" else ["python3", "-m", "pip", "install", wheel_path]
    subprocess.run(install_command, check=True)



def main():
    # clone onnxruntime_genai repo
    clone_repo(onnxruntime_genai_repo, onnxruntime_genai_dir)
    # clone onnxruntime repo
    clone_repo(onnxruntime_repo, onnxruntime_dir)
    # check for cuda
    use_cuda = check_cuda()
    # build onnxruntime
    if platform.system() == "Windows":
        build_onnxruntime_windows(onnxruntime_dir, use_cuda)
    else:
        build_onnxruntime_unix(onnxruntime_dir, use_cuda)
    # copy the build files
    if platform.system() == "Windows":
        copy_built_onnxruntime_windows(onnxruntime_dir, ort_dir)
    else:
        copy_built_onnxruntime_unix(onnxruntime_dir, ort_dir)
    # build onnxruntime-genai
    build_onnxruntime_genai(onnxruntime_genai_dir, ort_home=ort_dir)
    # install the wheel
    install_wheel(onnxruntime_genai_dir)




if __name__ == "__main__":
    main()
