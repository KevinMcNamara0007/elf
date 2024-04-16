import io

import pytesseract
from PIL import Image
from fastapi import HTTPException


async def extract_text_from_image(img):
    try:
        image_bytes = await img.read()
        image = Image.open(io.BytesIO(image_bytes))

        # convert image to grayscale
        image = image.convert("L")

        # Extract text using preprocessed image and tesseract
        text = pytesseract.image_to_string(image, lang="eng")
        return text
    except Exception as exc:
        HTTPException(status_code=500, detail=f"Could not extract image: {exc}")
