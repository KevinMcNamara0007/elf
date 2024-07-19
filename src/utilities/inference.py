import json
import math

import numpy as np
import requests
from fastapi import HTTPException
from src.utilities.general import (classifications, CONTEXT_WINDOW, tokenizer, classifier)


async def load_model(key):
    """
    return a preloaded model given a key.
    :param key:
    :return:
    """
    return classifications.get(key)["Link"]


async def classify_prompt(prompt, max_len=100, text=False):
    """
    Get a classification class from a given prompt.
    :param prompt: The prompt to classify.
    :param max_len: Maximum sequence length (default: 100).
    :param text: If True, the text is returned as a string.
    :return: Index of the predicted class.
    """
    try:
        # Tokenize prompt
        input_tokens = tokenizer.texts_to_sequences([prompt])

        # Pad input sequence
        max_seq_length = max_len
        input_tokens_padded = np.zeros((1, max_seq_length), dtype=np.float32)
        input_tokens_padded[:, :len(input_tokens[0])] = input_tokens[0][:max_seq_length]

        # Perform inference
        inputs = {classifier.get_inputs()[0].name: input_tokens_padded}
        prediction = classifier.run(None, inputs)[0]  # Retrieve the first output (assuming single output)

        # Process prediction (assuming binary classification)
        binary_prediction = [0 if x < 0.5 else 1 for x in prediction[0]]  # Assuming the first batch element

        # Check for ambiguity and return result
        if binary_prediction.count(1) != 1:
            return 1  # Fallback to a default class (e.g., safe choice)

        # Return index of the predicted class
        if not text:
            return binary_prediction.index(1)
        # Return category of the predicted class
        return classifications.get(binary_prediction.index(1))["Category"]

    except Exception as exc:
        # Handle any exceptions during inference
        raise HTTPException(status_code=500, detail=str(exc))


async def fetch_llama_cpp_response(messages, temperature, key, input_tokens=4000):
    try:
        expert = await load_model(key)
        payload = {
            "prompt": json.dumps(messages),
            "temperature": temperature,
            "n_predict": int(CONTEXT_WINDOW) - input_tokens,
            "stop": [
                "<|im_end|>"
            ]
        }
        expert_response = requests.post(expert, json=payload)
        expert_response.raise_for_status()
        return expert_response.json()
    except Exception as exc:
        print(str(exc))
        raise HTTPException(status_code=500, detail=f"Could not fetch response from llama.cpp: {exc}")

# async def audio_transcription(audiofile):
#     try:
#         with open(audiofile.filename, "wb+") as infile:
#             infile.write(await audiofile.read())
#         transcription = stt_pipe(audiofile.filename)["text"].strip()
#         os.remove(audiofile.filename)
#         return transcription
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=f"Could not fetch response from transcriber: {exc}")


# async def image_processing(prompt, image):
#     try:
#         messages = [
#             {"role": "user", "content": f"<|image_1|>\n{prompt}"}
#         ]
#         prompt = vision_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = vision_processor(prompt, [image], return_tensors="pt").to(device)
#         generation_args = {
#             "max_new_tokens": 2000,
#             "temperature": 0.0,
#             "do_sample": False
#         }
#         generate_ids = vision_model.generate(**inputs, eos_token_id=vision_processor.tokenizer.eos_token_id,
#                                              **generation_args)
#         generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
#         response = \
#             vision_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         return response
#     except Exception as exc:
#         print(str(exc))
#         raise HTTPException(status_code=500, detail=f"Error in image vision: {exc}")

# async def create_audio_from_transcription(transcript):
#     file_name = f"{str(int(time.time()))}.wav"
#     tts_model.tts_to_file(
#         text=transcript,
#         file_path=file_name
#     )
#     return file_name
