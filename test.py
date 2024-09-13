import glob
import importlib.util
import os

ort_package = importlib.util.find_spec("onnxruntime")
ort_package_path = ort_package.submodule_search_locations[0]
ort_lib_path = glob.glob(os.path.join(ort_package_path, "capi", "libonnxruntime*.dylib"))
print(ort_lib_path)
