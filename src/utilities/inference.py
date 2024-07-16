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


async def fetch_expert_response(messages, temperature, key, max_tokens=CONTEXT_WINDOW):
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


async def fetch_llama_cpp_response(rules, messages, temperature, key, max_tokens=CONTEXT_WINDOW):
    prompt = rules
    for message in messages:
        if message["role"] == "assistant":
            line = f"<|start_header_id|>system<|end_header_id|>\n\n{message['content']}<|eot_id|>"
        else:
            line = f"<|start_header_id|>user<|end_header_id|>\n\n{message['content']}<|eot_id|>"
        prompt += line
    try:
        expert = await load_model(key)
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": [
                "<|start_header_id|>user", "<|start_footer_id|>", "<|end_user|>", "<|start_header_id|>assistant",
                "<|eot_id|>"
            ]
            # Note the corrected temperature value
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
