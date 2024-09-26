import math
import httpx
import numpy as np
import requests
from fastapi import HTTPException
from src.utilities.general import (classifications, tokenizer, classifier,
                                   NUMBER_OF_SERVERS, CHATML_TEMPLATE, extract_port_from_url,
                                   LLM_TIMEOUT, ServerManager)

# Initialize the server manager with the number of servers
SERVER_MANAGER = ServerManager(NUMBER_OF_SERVERS)

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

async def fetch_llama_cpp_response(rules, messages, temperature, key, top_k=40, top_p=0.95):
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
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(expert_url, json=payload)
            response.raise_for_status()
            port = extract_port_from_url(expert_url)
            # Increment call count using the server manager
            for server in SERVER_MANAGER.servers:
                if server.port == port:
                    SERVER_MANAGER.increment_call_count(SERVER_MANAGER.servers.index(server))
            CURRENT_BALANCER_SELECTION = (CURRENT_BALANCER_SELECTION + 1) % int(BALANCER_MAX_OPTION)
            return response.json()
    except Exception as exc:
        print(f"Response Error: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Could not fetch response from llama.cpp: {exc.args}")

async def fetch_pro(prompt, output_tokens, key):
    global BALANCER_MAX_OPTION
    global CURRENT_BALANCER_SELECTION
    try:
        llama = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|start_header_id|>user<|end_header_id|><|eot_id|>assistant"
        expert_urls = load_model(key)
        payload = {
            "prompt": llama,
            "stream": False,
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
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(expert_url, json=payload)
            response.raise_for_status()
            port = extract_port_from_url(expert_url)
            for server in SERVER_MANAGER.servers:
                if server.port == port:
                    SERVER_MANAGER.increment_call_count(SERVER_MANAGER.servers.index(server))
            CURRENT_BALANCER_SELECTION = (CURRENT_BALANCER_SELECTION + 1) % int(BALANCER_MAX_OPTION)
            return response.json()
    except Exception as exc:
        print(f"Response Error: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Could not fetch response from llama.cpp: {exc.args}")

async def fetch_llama_cpp_response_stream(rules, messages, temperature, key, top_k=40, top_p=0.95):
    global BALANCER_MAX_OPTION
    global CURRENT_BALANCER_SELECTION
    try:
        prompt = convert_to_chat_template(rules, messages, template=CHATML_TEMPLATE)
        expert_urls = load_model(key)
        payload = {
            "prompt": prompt,
            "stream": True,  # Enable streaming in the request
            "temperature": temperature,
            "stop": STOP_SYMBOLS,
            "top_k": top_k,
            "top_p": top_p,
        }

        expert_url = f"{expert_urls[CURRENT_BALANCER_SELECTION]}/completion"
        port = extract_port_from_url(expert_url)
        CURRENT_BALANCER_SELECTION = (CURRENT_BALANCER_SELECTION + 1) % int(BALANCER_MAX_OPTION)

        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(expert_url, json=payload)
            response.raise_for_status()

            for server in SERVER_MANAGER.servers:
                if server.port == port:
                    SERVER_MANAGER.increment_call_count(SERVER_MANAGER.servers.index(server))

            # Streaming response chunks
            async for chunk in response.aiter_text():
                yield chunk
    except Exception as exc:
        print(f"Response Error: {str(exc)}")
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

async def call_openai(prompt: str, api_key: str):
    """
    Call the OpenAI API with the provided prompt and API key.

    :param prompt: The prompt for the LLM.
    :param api_key: The OpenAI API key.
    :return: The response from the OpenAI LLM.
    """
    openai_url = "https://api.openai.com/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(openai_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI: {exc}")

async def call_claude(prompt: str, api_key: str):
    """
    Call the Claude API with the provided prompt and API key.

    :param prompt: The prompt for the LLM.
    :param api_key: The Claude API key.
    :return: The response from the Claude LLM.
    """
    claude_url = "https://api.anthropic.com/v1/complete"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "max_tokens_to_sample": 100,
        "stop_sequences": ["\n"],
        "temperature": 0.7,
        "model": "claude-v1"
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(claude_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=500, detail=f"Error calling Claude: {exc}")
