import subprocess
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
import requests

app = FastAPI()

CONTEXT_WINDOW = "8196"
PORT = "8080"
LLAMA_CPP_ENDPOINT = f"http://127.0.0.1:{PORT}/completion"
LLAMA_CPP_PATH = "efs/frameworks/llama.cpp/build/bin/Release/llama-server.exe"  # Replace with actual path to llama-server
MODEL_PATH = "efs/models/mistral-7b-instruct-v0.2.Q2_K.gguf"  # Replace with the actual path to your model

llama_cpp_process = None


def start_llama_cpp():
    global llama_cpp_process
    llama_cpp_process = subprocess.Popen([LLAMA_CPP_PATH, "-m", MODEL_PATH, "-c", CONTEXT_WINDOW, "-p", PORT])


def stop_llama_cpp():
    global llama_cpp_process
    if llama_cpp_process:
        llama_cpp_process.terminate()
        llama_cpp_process.wait()


@app.on_event("startup")
async def startup_event():
    start_llama_cpp()


@app.on_event("shutdown")
async def shutdown_event():
    stop_llama_cpp()


@app.post("/infer")
def infer_text(
        text: str = Form()
):
    payload = {
        "prompt": text,
        "n_predict": -1,
        "temperature": .5
    }

    try:
        response = requests.post(LLAMA_CPP_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return JSONResponse(content={"result": result})
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with llama.cpp: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
