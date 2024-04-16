from fastapi import APIRouter, status, Form, UploadFile, File

from src.services.images import extract_text_from_image

images_router = APIRouter(
    prefix="/Images",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["Image Endpoints"]
)


@images_router.post("/text_extraction",
                    status_code=status.HTTP_200_OK,
                    description="Returns a list of the available models.")
async def fetch_text_extraction(
        image: UploadFile = File(description="Image to extract text from")
):
    return await extract_text_from_image(image)
