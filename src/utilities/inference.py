import os
import time
from fastapi import HTTPException
from src.utilities.general import classifier, tokenizer, classifications, context_window, stt_pipe#, tts_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


async def load_model(key):
    """
    return a preloaded model given a key.
    :param key:
    :return:
    """
    return classifications.get(key)["Model"]


async def classify_prompt(prompt, c_tokenizer=tokenizer, c_model=classifier, max_len=100):
    """
    Get a classification class from a given prompt.
    :param prompt:
    :param c_tokenizer:
    :param c_model:
    :param max_len:
    :return:
    """
    try:
        # Tokenize the prompt
        seq = c_tokenizer.texts_to_sequences([prompt])
        # Pad the sequence
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        # Predict the category
        prediction = c_model.predict(padded)[0]
        # Convert predictions to binary
        binary_prediction = [0 if x < .49 else 1 for x in prediction]
        # Check for ambiguity and go with safe model if found
        if binary_prediction.count(1) > 1 or binary_prediction.count(1) < 1:
            return 1
        # Return index of max value in the list of predictions
        return binary_prediction.index(max(binary_prediction))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=exc)


async def fetch_expert_response(messages, temperature, key, max_tokens=context_window):
    """
    Call one of the models depending on the given key, pass the messages and temperature to use the model in its
    inference mode. Max tokens refers to the output tokens.
    :param max_tokens:
    :param messages:
    :param temperature:
    :param key:
    :return:
    """
    try:
        expert = await load_model(key)
        return expert.create_chat_completion(
            messages=messages,
            temperature=temperature,
            # max number of tokens to be generated
            max_tokens=max_tokens
        )
    except ValueError as val_err:
        print(val_err)
        return {
            "model": "efs/models/mistral-7b-instruct-v0.2.Q2_K.gguf",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Input tokens exceed model limit"
                    },
                    "finish_reason": "exceed"
                }
            ],
            "usage": {
                "prompt_tokens": 8196,
                "completion_tokens": 0,
                "total_tokens": 8196
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch response from model: {exc}")


async def audio_transcription(audiofile):
    try:
        with open(audiofile.filename, "wb+") as infile:
            infile.write(await audiofile.read())
        transcription = stt_pipe(audiofile.filename)["text"].strip()
        os.remove(audiofile.filename)
        return transcription
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch response from transcriber: {exc}")


# async def create_audio_from_transcription(transcript):
#     file_name = f"{str(int(time.time()))}.wav"
#     tts_model.tts_to_file(
#         text=transcript,
#         file_path=file_name
#     )
#     return file_name

