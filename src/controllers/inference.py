import json
from fastapi import APIRouter, status, Form, HTTPException, UploadFile, File
from starlette.responses import FileResponse
from src.services.inference import get_all_models, get_expert_response, prompt_classification
from src.utilities.general import file_cleanup
# from src.utilities.inference import audio_transcription#, create_audio_from_transcription

inference_router = APIRouter(
    prefix="/Inference",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["Inference Endpoints"]
)


@inference_router.get("/all_models", status_code=status.HTTP_200_OK,
                      description="Returns a list of the available models.")
async def fetch_all_models():
    return await get_all_models()


@inference_router.post("/ask_an_expert", status_code=status.HTTP_200_OK, description="Ask any question.")
async def ask_an_expert(
        messages: str = Form(default=None, description="Chat style prompting"),
        prompt: str = Form(default=None, description="The prompt you want answered."),
        temperature: float = Form(default=0.05, description="Temperature of the model."),
        rules: str = Form(default="You are a friendly virtual assistant. Your role is to answer the user questions and follow their instructions. Be concise and accurate.",
                          description="Rules of the model."),
        top_k: int = Form(default=40),
        top_p: float = Form(default=0.95)
):
    history = []
    # Validate and parse the messages
    if messages:
        try:
            history.extend(json.loads(messages))
            if not isinstance(history, list):
                raise HTTPException(status_code=400, detail="Messages must be a list of dictionaries.")
            for message in history:
                if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                    raise HTTPException(status_code=400, detail="Each message must be a dictionary with 'role' and 'content' keys.")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format for messages. {exc}")

    if prompt:
        history.append({"role": "User", "content": prompt})
    if not messages and not prompt:
        raise HTTPException(status_code=400, detail="Provide Messages, Prompt or both.")

    return await get_expert_response(
        rules=rules,
        messages=history,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


@inference_router.post("/classify", status_code=status.HTTP_200_OK, description="Classify your prompt.")
async def determine_expert(
        prompt: str = Form()
):
    return await prompt_classification(prompt)


# @inference_router.post("/stt", description="Get transcription for audio file.")
# async def generate_transcription(
#         audiofile: UploadFile = File(description="The file you would like transcribed")
# ):
#     return await audio_transcription(audiofile)


# @inference_router.post("/tts", description="Get transcription for audio file.")
# async def generate_speech(
#         background_tasks: BackgroundTasks,
#         transcript: str = Form(description="The text to be converted to audio")
# ):
#     file_name = await create_audio_from_transcription(transcript)
#     # background_tasks.add_task(file_cleanup, file_name)
#     return FileResponse(file_name)
