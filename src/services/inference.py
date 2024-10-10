import json
import time
from datetime import datetime
import os

import requests

from src.utilities.general import llama_manager, classifications
from src.utilities.inference import classify_prompt

# Load environment variables
CHATML_TEMPLATE = os.getenv("CHATML_TEMPLATE")
LLAMA3_TEMPLATE = os.getenv("LLAMA3_TEMPLATE")
CHAT_TEMPLATE = LLAMA3_TEMPLATE if "llama" in os.getenv("general").lower() else CHATML_TEMPLATE

STOP_SYMBOLS = [
    "</s>", "Llama:", "User:", "<|end|>", "<|eot_id|>", "<|end_of_text|>",
    "<|eom_id|>", "<|im_end|>", "<|EOT|>", "<|END_OF_TURN_TOKEN|>",
    "<|end_of_turn|>", "<|endoftext|>", "assistant", "user", "<|end_header_id|>"
]


# Template constructors for chat formatting
def format_llama3(messages):
    return "\n".join(
        f"{'<|start_header_id|>assistant<|end_header_id|>' if 'user' not in message.role.lower() else '<|start_header_id|>user<|end_header_id|>'}\n\n{message.content}<|eot_id|>\n" for message in messages
    )


def format_chatml(messages):
    return "\n".join(
        f"<|im_start|>{'assistant' if 'user' not in message.role.lower() else 'user'}\n{message.content}<|im_end|>"
        for message in messages
    )


def convert_to_chat_template(rules, messages, template=CHAT_TEMPLATE):
    if template == CHATML_TEMPLATE:
        return f"<|im_start|>system\n{rules}<|im_end|>\n{format_chatml(messages)}\nassistant"
    return (f'<|start_header_id|>system<|end_header_id|>'
            f"Today is {datetime.today().strftime('%Y-%m-%d')}{rules}<|eot_id|>\n") + format_llama3(
        messages) + "\n<|start_header_id|>assistant<|end_header_id|>"


# Core method for fetching a response from the server
async def get_expert_response(rules, messages, temperature=0.8, top_k=40, top_p=0.95):
    # Classify the last message to fetch the correct key
    key = await classify_prompt(messages[-1].content)
    print(f"Classification: {classifications[key]}")

    # Generate prompt and call llama-server
    prompt = convert_to_chat_template(rules, messages, CHAT_TEMPLATE)
    response = await llama_manager.call_llama_server({
        "prompt": prompt,
        "n_predict": -1,
        "stop": STOP_SYMBOLS,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": False,
    })

    return llama_response_formatter(response)


# General response formatter
def llama_response_formatter(response):
    prompt_tokens = int(response['tokens_evaluated'])
    completion_tokens = int(response['tokens_predicted'])

    return {
        'usage': {
            'total_tokens': prompt_tokens + completion_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
        },
        'choices': [{
            'message': {'content': response['content'].replace("<|end_header_id|>", "")},
            'finish_reason': 'length' if response.get('stopped_limit') else 'stop',
        }],
        'timings': response['timings']
    }


# Fetch response for pro version (simplified prompt)
async def get_pro_response(prompt):
    key = await classify_prompt(prompt)
    print(f"Classification: {classifications[key]}")
    llama_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>assistant"
    response = await llama_manager.call_llama_server({
        "prompt": llama_prompt,
        "stream": False,
        "temperature": 0.8,
        "stop": ["</s>", "<|end|>", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|EOT|>", "<|END_OF_TURN_TOKEN|>", "<|end_of_turn|>", "<|endoftext|>", "assistant", "user"],
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
        "api_key": "",
    })
    return llama_response_formatter(response)


async def get_pro_response_stream(prompt, output_tokens):
    """
    Fetches response from llama-server and recalls if generation is truncated. Returns the full response.
    """
    key = await classify_prompt(prompt)
    prompt1 = (f"Instructions: "
               f"1. For the following user ask {prompt} you will clarify the ask."
               f"2. List the requirements and steps needed to complete this task fully.")
    response1 = ""
    async for chunk in fetch_pro_stream(prompt=prompt1,output_tokens=output_tokens, key=key):
        arr = chunk.split(': ', 1)[1]
        data_dict = json.loads(arr)
        content = data_dict.get('content')
        response1 += content
        yield content
    prompt2 = ("Instructions:"
               f"1. Fulfill the requirements listed by producing a response that accomplishes all of them: {response1}"
               )
    response2 = ""
    yield "\n!Final!\n"
    time.sleep(3)
    async for chunk in fetch_pro_stream(prompt=prompt2,output_tokens=output_tokens, key=key):
        arr = chunk.split(': ', 1)[1]
        data_dict = json.loads(arr)
        content = data_dict.get('content')
        response2 += content
        yield content


async def fetch_pro_stream(prompt, output_tokens, key):
    llama = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|start_header_id|>user<|end_header_id|><|eot_id|>assistant"
    payload = {
        "prompt": llama,
        "stream": True,  # Enable streaming
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
    expert_url = f"http://127.0.0.1:8001/completion"
    response = requests.post(
        expert_url,
        json=payload,
        stream=True
    )
    response.raise_for_status()
    for chunk in response.iter_lines():
        if chunk:
            yield chunk.decode('utf-8')


# Stream expert response
async def get_expert_response_stream(rules, messages, temperature=0.05, top_k=40, top_p=0.95):
    key = await classify_prompt(messages[-1].content)
    print(f"Classification: {classifications[key]}")

    payload = {
        "rules": rules,
        "messages": messages,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "key": key,
        "stream": True,
    }

    async for chunk in llama_manager.call_llama_server_stream(payload):
        yield chunk  # Stream output chunk by chunk


# Classify a given prompt
def prompt_classification(prompt):
    return classify_prompt(prompt, text=True)
