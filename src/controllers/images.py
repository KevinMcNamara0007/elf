# from fastapi import APIRouter, status, Form, UploadFile, File
#
# from src.services.images import extract_text_from_image, vision_for_images
#
# images_router = APIRouter(
#     prefix="/Images",
#     responses={
#         200: {"description": "Successful"},
#         400: {"description": "Bad Request"},
#         500: {"description": "Internal Server Error"},
#     },
#     tags=["Image Endpoints"]
# )
#
#
# @images_router.post("/text_extraction",
#                     status_code=status.HTTP_200_OK,
#                     description="Extract text from image.")
# async def fetch_text_extraction(
#         image: UploadFile = File(description="Image to extract text from")
# ):
#     return await extract_text_from_image(image)
#
#
# @images_router.post("/image_vision", status_code=status.HTTP_200_OK,
#                     description="Performs vision inference on image.")
# async def fetch_image_vision(
#         image: UploadFile = File(description="Image to perform vision inference on."),
#         prompt: str = File(
#             default="What is shown on this image?",
#             description="What the model should do with the image."
#         )
# ):
#     return await vision_for_images(prompt, image)
