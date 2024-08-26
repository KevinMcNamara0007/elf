import os
import torch
import transformers
from torch.onnx import export
from PIL import Image
from huggingface_hub import snapshot_download
import onnxruntime as ort

# Ensure FlashAttention2 is disabled if the package isn't available
os.environ['USE_FLASHATTN'] = '0'

# Base directory for storing models
base_model_dir = "/opt/models/microsoft/Phi-3-vision-128k-instruct"
onnx_output_dir = "/opt/models/ONNX"

# Ensure output directories exist
os.makedirs(onnx_output_dir, exist_ok=True)


def download_repo(model_name, model_dir):
    """Download the entire repository from Hugging Face."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

        # Download the entire repository snapshot
        print(f"Downloading the repository {model_name} to {model_dir}...")
        try:
            snapshot_download(repo_id=model_name, local_dir=model_dir)
        except Exception as e:
            print(f"Error downloading repository {model_name}: {e}")
            return


def load_and_convert_to_onnx(model_dir, onnx_dir):
    """Load a model using transformers and convert it to ONNX format."""
    onnx_model_path = os.path.join(onnx_dir, "Phi-3-vision-128k-instruct.onnx")
    processor = transformers.AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    if not os.path.exists(onnx_model_path):
        # Load the model with float32 instead of bfloat16
        dtype = torch.float32  # Change from bfloat16 to float32
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="auto",
            _attn_implementation="eager",
            trust_remote_code=True,
        )
        # Export the model to ONNX
        dummy_input = processor(text="Example input", return_tensors="pt").to("cpu")
        onnx_model_path = os.path.join(onnx_dir, "Phi-3-vision-128k-instruct.onnx")
        export(model=model, args=(dummy_input["input_ids"],), f=onnx_model_path, opset_version=14)

        print(f"Model converted to ONNX and saved to {onnx_model_path}")
    return onnx_model_path, processor


def setup_model(model_name, model_dir, onnx_dir):
    """Setup and load the model components after downloading and converting the repo."""
    download_repo(model_name, model_dir)

    # Convert model from transformers to ONNX
    onnx_model_path, processor = load_and_convert_to_onnx(model_dir, onnx_dir)

    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)

    return ort_session, processor


def process_image_with_prompt(onnx_session, processor, image_path, prompt):
    """Process the image with a text prompt using the ONNX model."""
    image = Image.open(image_path)
    inputs = processor(images=image, text=prompt, return_tensors="np")
    print(inputs.keys())
    onnx_inputs = {onnx_session.get_inputs()[0].name: inputs["input_ids"]}
    onnx_outputs = onnx_session.run(None, onnx_inputs)
    image_output = onnx_outputs[0]

    return image_output


# Example usage
model_name = "microsoft/Phi-3-vision-128k-instruct"
vision_model, feature_processor = setup_model(model_name, base_model_dir, onnx_output_dir)

# Now you can use ort_session and feature_extractor for inference
image_path = "efs/space_dog.jpg"
prompt = "<|user|>\n<|image_1|>\nWhat do you see in this image?<|end|>\n"
image_output = process_image_with_prompt(vision_model, feature_processor, image_path, prompt)

print("Image Output:", image_output)
