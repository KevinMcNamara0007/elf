import tempfile
import onnxruntime_genai as og
from src.utilities.inference import vision_and_text_inference


async def vision_for_images(prompt, image=None):
    """
    Processes images and prompt to perform vision inference.

    Args:
        prompt (str): The text prompt for the model.
        image (UploadFile): Image being passed to the model.

    Returns:
        str: Model response.
    """
    loaded_image = None
    if image:
        # Use a temporary file to store the image data
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
            image_bytes = await image.read()
            temp_file.write(image_bytes)
            temp_file.flush()
            loaded_image = og.Images.open(temp_file.name)

    # Add image tags to the prompt
    user_prompt = f"<|user|>\n{'<|image_1|>' if loaded_image else ''}\n{prompt}\n<|end|>\n<|assistant|>\n"
    response = vision_and_text_inference(user_prompt, loaded_image)

    return response
