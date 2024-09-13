import os
from io import BytesIO
from PIL import Image
import numpy as np
from fastapi import UploadFile, HTTPException
import onnxruntime_genai as og

from src.utilities.inference import vision_and_text_inference

# List of supported image formats for Phi-3 vision model
SUPPORTED_FORMATS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}


async def preprocess_image(image_bytes: bytes, target_size=(224, 224)):
    # Load the image from bytes
    image = Image.open(BytesIO(image_bytes))

    # Resize the image
    image = image.resize(target_size)

    # Convert to RGB
    image = image.convert('RGB')

    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0

    return image_array


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
        # Read image bytes
        image_bytes = await image.read()

        # Get the file extension from the uploaded image
        file_extension = image.filename.split('.')[-1].lower()

        if file_extension not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400,
                                detail="Unsupported image format. Supported formats are: jpg, jpeg, png, bmp, gif.")

        # Preprocess the image
        preprocessed_image = await preprocess_image(image_bytes)

        # Define the filename with the original extension
        temp_filename = f"temp_image.{file_extension}"

        # Save preprocessed image to the current directory with its original extension
        preprocessed_pil_image = Image.fromarray((preprocessed_image * 255).astype(np.uint8))
        preprocessed_pil_image.save(temp_filename)

        # Open the saved image
        loaded_image = og.Images.open(temp_filename)

        # Remove the temporary file after loading the image
        os.remove(temp_filename)

    # Add image tags to the prompt
    user_prompt = f"<|user|>\n{'<|image_1|>' if loaded_image else ''}\n{prompt}\n<|end|>\n<|assistant|>\n"

    try:
        response = vision_and_text_inference(user_prompt, loaded_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during vision inference.")

    return response
