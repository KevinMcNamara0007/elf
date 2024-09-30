import json
import math
import httpx
import numpy as np
import requests
from fastapi import HTTPException
from src.utilities.general import (classifications, CONTEXT_WINDOW, tokenizer, classifier,
                                   NUMBER_OF_SERVERS, CHAT_TEMPLATE, LLAMA3_TEMPLATE, CHATML_TEMPLATE)


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
STOP_SYMBOLS = []


def llama3_template(messages):
    transcript = ""
    for message in messages:
        transcript += f"{'Llama' if 'user' not in message.role.lower() else 'User'}: {message.content}\n\n"
    return transcript


def chatml_template(messages):
    transcript = ""
    for message in messages:
        transcript += f"<|im_start|>{'assistant' if 'user' not in message.role.lower() else 'user'}\n {message.content}<|im_end|>\n"
    return transcript


def convert_to_chat_template(rules, messages, template=CHATML_TEMPLATE):
    global STOP_SYMBOLS
    if template == CHATML_TEMPLATE:
        STOP_SYMBOLS = ["</s>",
                        "<|end|>",
                        "<|eot_id|>",
                        "<|end_of_text|>",
                        "<|im_end|>",
                        "<|EOT|>",
                        "<|END_OF_TURN_TOKEN|>",
                        "<|end_of_turn|>",
                        "<|endoftext|>",
                        "assistant",
                        "user"
                        ]
        return f"<|im_start|>system\n{rules}<|im_end|>\n{chatml_template(messages)}\nassistant"
    else:
        STOP_SYMBOLS = ["</s>", "Llama:", "User:"]
        return rules + llama3_template(messages) + "Llama:"


async def fetch_llama_cpp_response(rules, messages, temperature, key, top_k=40, top_p=.95):
    """
    Sends request to llama-server for inference
    """
    global BALANCER_MAX_OPTION
    global CURRENT_BALANCER_SELECTION
    try:
        expert_urls = load_model(key)
        payload = {
            "prompt": convert_to_chat_template(rules, messages, template=CHATML_TEMPLATE),
            "stream": False,
            "n_predict": -1,
            "temperature": temperature,
            "stop": STOP_SYMBOLS,
            "top_k": top_k,
            "top_p": top_p,
        }

        expert_url = f"{expert_urls[CURRENT_BALANCER_SELECTION]}/completion"
        CURRENT_BALANCER_SELECTION = (CURRENT_BALANCER_SELECTION + 1) % BALANCER_MAX_OPTION
        async with httpx.AsyncClient(timeout=300) as client:
            expert_response = await client.post(expert_url, json=payload)
            expert_response.raise_for_status()
            response_data = expert_response.json()
        return response_data
    except httpx.RequestError as e:
        print("Network Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Network error while fetching response from llama.cpp: {e}")
    except Exception as exc:
        print("Response Error:", str(exc))
        raise HTTPException(status_code=500, detail=f"Could not fetch response from llama.cpp: {exc.args}")


async def fetch_pro(prompt, output_tokens, key):
    """
    Sends request to llama-server for inference
    """
    global BALANCER_MAX_OPTION
    global CURRENT_BALANCER_SELECTION
    llama = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|start_header_id|>user<|end_header_id|><|eot_id|>assistant"
    try:
        expert_urls = load_model(key)
        payload = {
            "prompt": llama,
            "stream": False,
            "n_predict": math.ceil(output_tokens),
            "temperature": 0.8,
            "stop":
                ["</s>",
                 "<|end|>",
                 "<|eot_id|>",
                 "<|end_of_text|>",
                 "<|im_end|>",
                 "<|EOT|>",
                 "<|END_OF_TURN_TOKEN|>",
                 "<|end_of_turn|>",
                 "<|endoftext|>",
                 "assistant",
                 "user"],
            "repeat_last_n":0,
            "repeat_penalty":1,
            "penalize_nl":False,
            "top_k":0,
            "top_p":1,
            "min_p":0.05,
            "tfs_z":1,
            "typical_p":1,
            "presence_penalty":0,
            "frequency_penalty":0,
            "mirostat":0,
            "mirostat_tau":5,
            "mirostat_eta":0.1,
            "grammar":"",
            "n_probs":0,
            "min_keep":0,
            "image_data":[],
            "cache_prompt":False,
            "api_key":""
        }

        expert_url = f"{expert_urls[CURRENT_BALANCER_SELECTION]}/completion"
        CURRENT_BALANCER_SELECTION = (CURRENT_BALANCER_SELECTION + 1) % BALANCER_MAX_OPTION
        async with httpx.AsyncClient(timeout=500) as client:
            expert_response = await client.post(expert_url, json=payload)
            expert_response.raise_for_status()
            response_data = expert_response.json()
        return response_data
    except httpx.RequestError as e:
        print(e)
        print("Network Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Network error while fetching response from llama.cpp: {e}")
    except Exception as exc:
        print("Response Error:", str(exc))
        raise HTTPException(status_code=500, detail=f"Could not fetch response from llama.cpp: {exc.args}")


async def fetch_llama_cpp_response_stream(rules, messages, temperature, key, top_k=40, top_p=0.95):
    """
    Sends request to llama-server for inference and streams the response back.
    """
    global BALANCER_MAX_OPTION
    global CURRENT_BALANCER_SELECTION
    try:
        prompt = convert_to_chat_template(rules, messages, template=CHATML_TEMPLATE)
        expert_urls = load_model(key)
        payload = {
            "prompt": prompt,
            "stream": True,  # Enable streaming in the request
            "n_predict": -1,
            "temperature": temperature,
            "stop": STOP_SYMBOLS,
            "top_k": top_k,
            "top_p": top_p,
        }

        expert_url = f"{expert_urls[CURRENT_BALANCER_SELECTION]}/completion"
        CURRENT_BALANCER_SELECTION = (CURRENT_BALANCER_SELECTION + 1) % BALANCER_MAX_OPTION
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", expert_url, json=payload) as response:
                async for chunk in response.aiter_text():
                    yield chunk  # Yield each chunk as it is received
    except httpx.RequestError as e:
        print("Network Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Network error while fetching response from llama.cpp: {e}")
    except Exception as exc:
        print("Response Error:", str(exc))
        raise HTTPException(status_code=500, detail=f"Could not fetch response from llama.cpp: {exc.args}")


def get_free_url(urls):
    """
    Iterates through all available servers until a free server is available. Returns the free url.
    """
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


async def fetch_pro_stream(prompt, output_tokens, key):
    global BALANCER_MAX_OPTION
    global CURRENT_BALANCER_SELECTION
    try:
        llama = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|start_header_id|>user<|end_header_id|><|eot_id|>assistant"
        expert_urls = load_model(key)
        payload = {
            "prompt": llama,
            "stream": True,  # Enable streaming
            "n_predict": math.ceil(output_tokens),
            "temperature": 0.8,
            "stop": ["</s>", "<|end|>", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|EOT|>",
                     "<|END_OF_TURN_TOKEN|>", "<|end_of_turn|>", "<|endoftext|>", "assistant", "user"],
            "repeat_last_n": 0,
            "repeat_penalty": 1,
            "penalize_nl": False,
            "top_k": 0,
            "top_p": 1,
            "min_p": 0.05,
            "tfs_z": 1,
            "typical_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "mirostat": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
            "grammar": "",
            "n_probs": 0,
            "min_keep": 0,
            "image_data": [],
            "cache_prompt": False,
            "api_key": ""
        }
        expert_url = f"{expert_urls[CURRENT_BALANCER_SELECTION]}/completion"
        response = requests.post(
            expert_url,
            json=payload,
            stream=True
        )
        response.raise_for_status()
        CURRENT_BALANCER_SELECTION = (CURRENT_BALANCER_SELECTION + 1) % BALANCER_MAX_OPTION
        for chunk in response.iter_lines():
            if chunk:
                yield chunk.decode('utf-8')
    except Exception as exc:
        print(f"Response Error: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Could not fetch response from llama.cpp: {exc.args}")


