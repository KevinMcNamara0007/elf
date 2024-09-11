import os
import subprocess
import urllib.request
import tarfile
import shutil
import onnxruntime as ort

onnxruntime_genai_repo = "https://github.com/microsoft/onnxruntime-genai"
onnxruntime_release_url = "https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-osx-universal2-1.19.0.tgz"
onnxruntime_tar = "onnxruntime-osx-universal2-1.19.0.tar.gz"
onnxruntime_dir = "onnxruntime-osx-universal2-1.19.0"
ort_dir = "onnxruntime-genai/ort"


def clone_repo(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print(f"{repo_dir} already exists. Skipping clone")


def download_release(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(e)
            exit(1)
    else:
        print(f"{filename} already exists. Skipping download")


def extract_tar(filename, extract_path):
    if not os.path.exists(extract_path):
        print(f"Extracting {filename} to {extract_path}")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(extract_path)
    else:
        print(f"{filename} already exists. Skipping extract")


def move_onnxruntime_files(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                print(f"{s} already exists. Skipping copy")
        else:
            shutil.copy2(s, d)


def copy_pip_installed_onnxruntime(dest_dir):
    ort_library_path = os.path.dirname(ort.__file__)
    ort_include_path = os.path.join(ort_library_path, "include")
    ort_lib_files = [f for f in os.listdir(ort_library_path) if f.endswith(".dylib")]
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for lib_file in ort_lib_files:
        shutil.copy2(os.path.join(ort_library_path, lib_file), os.path.join(dest_dir, "lib", lib_file))
    if os.path.exists(ort_include_path):
        shutil.copytree(ort_include_path, os.path.join(dest_dir, "include"))


def build_onnxruntime_genai(repo_dir, ort_home=None):
    os.chdir(repo_dir)
    build_command = ["python3", "build.py"]
    if ort_home:
        build_command.extend(["--ort_home", ort_home])
    subprocess.run(build_command, check=True)


def main():
    clone_repo(onnxruntime_genai_repo, "onnxruntime-genai")

    download_release(onnxruntime_release_url, onnxruntime_tar)

    extract_tar(onnxruntime_tar, onnxruntime_dir)

    move_onnxruntime_files(onnxruntime_dir, os.path.join("onnxruntime-genai", ort_dir))

    copy_pip_installed_onnxruntime(os.path.join(ort_dir, "lib"))

    build_onnxruntime_genai("onnxruntime-genai")


if __name__ == "__main__":
    main()
