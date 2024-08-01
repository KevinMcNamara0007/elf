import json
import math

import httpx
import numpy as np
import requests
from fastapi import HTTPException
from src.utilities.general import (classifications, CONTEXT_WINDOW, tokenizer, classifier, LLAMA_CPP_ENDPOINTS,
                                   NUMBER_OF_SERVERS)


def load_model(key):
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


BALANCER_MAX_OPTION = NUMBER_OF_SERVERS
CURRENT_BALANCER_SELECTION = 0


async def fetch_llama_cpp_response(rules, messages, temperature, key, input_tokens=4000, top_k=40, top_p=0.95):
    global BALANCER_MAX_OPTION
    global CURRENT_BALANCER_SELECTION
    try:
        expert_urls = load_model(key)
        payload = {
            "prompt": json.dumps(messages),
            "temperature": temperature,
            "n_predict": int(CONTEXT_WINDOW) - input_tokens,
            "system_prompt": rules,
            "top_k": top_k,
            "top_p": top_p,
            "stop": [
                "<|im_end|>",
                "</s>",
                "<end_of_turn>",
                '[{\"role\":'
            ]
        }

        expert_url = f"{expert_urls[CURRENT_BALANCER_SELECTION]}/completion"
        CURRENT_BALANCER_SELECTION = (CURRENT_BALANCER_SELECTION + 1) % BALANCER_MAX_OPTION
        async with httpx.AsyncClient(timeout=300) as client:
            expert_response = await client.post(expert_url, json=payload)
            expert_response.raise_for_status()
            response_data = expert_response.json()

        response_data['content'] = (
            response_data['content'].replace(' assistant', '').replace('assistant', '').replace('<|im_start|>', '').replace('<|im_end|>', '')
        )
        return response_data
    except Exception as exc:
        print("Response Error:", str(exc))
        raise HTTPException(status_code=500, detail=f"Could not fetch response from llama.cpp: {exc.args}")


def get_free_url(urls):
    try:
        free = None
        while not free:
            for url in urls:
                req = requests.get(f"{url}/health")
                response = req.json()
                print(response)
                free = None if response["status"] == "no slot available" else url
        return f"{free}/completion"
    except Exception as exc:
        print(str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

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
