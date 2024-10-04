import os

from src.utilities.general import llama_manager, classifications
from src.utilities.inference import classify_prompt


CHATML_TEMPLATE = os.getenv("CHATML_TEMPLATE")
LLAMA3_TEMPLATE = os.getenv("LLAMA3_TEMPLATE")
CHAT_TEMPLATE = LLAMA3_TEMPLATE if "llama" in os.getenv("general").lower() else CHATML_TEMPLATE


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


def convert_to_chat_template(messages, template=CHATML_TEMPLATE):
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
        return f"{chatml_template(messages)}\nassistant"
    else:
        STOP_SYMBOLS = ["</s>", "Llama:", "User:"]
        return llama3_template(messages) + "Llama:"


# Inject the classifier and tokenizer using Depends
async def get_expert_response(
        rules, messages, temperature=0.05, top_k=40, top_p=0.95
):
    """
    Fetches response from llama-server and recalls if generation is truncated. Returns the full response.
    """

    # Get the key by classifying the last message in the conversation
    key = await classify_prompt(messages[-1].content)

    print("Classification: {}".format(classifications[key]))

    # Fetch response from llama-server
    response = await llama_manager.call_llama_server(
        {
            "system_prompt": f"<|im_start|>system\n{rules}<|im_end|>" if CHAT_TEMPLATE == CHATML_TEMPLATE else rules,
            "prompt": convert_to_chat_template(messages, CHAT_TEMPLATE),
            "stop": STOP_SYMBOLS,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False
        }
    )
    return llama_response_formatter(response)


async def get_pro_response(prompt):
    """
    Fetches response from llama-server and recalls if generation is truncated. Returns the full response.
    """

    # Classify the prompt
    key = await classify_prompt(prompt)

    print("Classification: {}".format(classifications[key]))

    llama = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|start_header_id|>user<|end_header_id|><|eot_id|>assistant"

    # Fetch response from llama-server
    response = await llama_manager.call_llama_server({
        "prompt": llama,
        "stream": False,
        "n_predict": -1,
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
    })
    return llama_response_formatter(response)


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
            'message': {
                'content': response['content']
            },
            'finish_reason': 'length' if response['stopped_limit'] else 'stop',
        }],
        'timings': response['timings']
    }


async def get_expert_response_stream(
        rules, messages, temperature=0.05, top_k=40, top_p=0.95
):
    """
    Fetches response from llama-server and streams the result. Yields each chunk as it's received.
    """
    # Classify the last message
    key = await classify_prompt(messages[-1].content)

    print(f"Classification: {classifications.get(key)['Category']}")

    # Prepare payload for llama-server
    payload = {
        "rules": rules,
        "messages": messages,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "key": key,
    }

    # Call the async generator function and stream its output
    async for chunk in llama_manager.call_llama_server_stream(payload):
        yield chunk  # Yield each chunk as it is received


def prompt_classification(prompt):
    """
    Classifies a given prompt using the cnn classifier.
    """
    return classify_prompt(prompt, text=True)
