from fastapi import APIRouter, UploadFile, File, Body, HTTPException, status

from src.services.images import vision_for_images

images_router = APIRouter(
    prefix="/Images",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["Image Endpoints"]
)


@images_router.post("/image_vision", status_code=status.HTTP_200_OK,
                    description="Performs vision inference on image.")
async def fetch_image_vision(
        prompt: str = Body(...),
        image: UploadFile = File(...)
):
    return await vision_for_images(prompt, image)
