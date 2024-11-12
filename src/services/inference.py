import json
import re
from datetime import datetime
import os
from src.utilities.general import llama_manager, classifications, STREAM_PAYLOAD
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
        f"{'<|start_header_id|>assistant<|end_header_id|>' if 'user' not in message.role.lower() else '<|start_header_id|>user<|end_header_id|>'}\n\n{message.content}<|eot_id|>\n"
        for message in messages
    ) if type(messages) == list else f"<|start_header_id|>user<|end_header_id|>\n{messages}<|eot_id|>\n"


def format_chatml(messages):
    return "\n".join(
        f"<|im_start|>{'assistant' if 'user' not in message.role.lower() else 'user'}\n{message.content}<|im_end|>"
        for message in messages
    ) if type(messages) == list else f"<|im_start|>user\n{messages}<|im_end|>"


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
        "penalize_nl": True,
        "repeat_last_n": 0,
        "min_keep": 0
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
    llama_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    response = await llama_manager.call_llama_server({
        "prompt": llama_prompt,
        "stream": False,
        "temperature": 0.8,
        "stop": STOP_SYMBOLS,
        "repeat_last_n": 0,
        "repeat_penalty": 1,
        "penalize_nl": True,
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


async def get_pro_response_stream(prompt):
    """
    Fetches response from llama-server in chunks, handling truncation and yielding a final output response.
    """
    key = await classify_prompt(prompt)
    print(f"Classification: {classifications[key]}")
    payload = STREAM_PAYLOAD
    response1 = ""
    llama_prompt1 = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    # Initial response for clarification
    payload.update(
        {"prompt": llama_prompt1}
    )
    payload.update({"stop": STOP_SYMBOLS})

    # Stream the initial clarification response
    async for chunk in llama_manager.call_llama_server_stream(payload):
        try:
            arr = chunk.split(': ', 1)[1]
            data_dict = json.loads(arr)
            content = data_dict.get('content')
            response1 += content
            print(content)
            yield content
        except (json.JSONDecodeError, IndexError):
            print("Failed to parse chunk:", chunk)


# Stream expert response
async def get_expert_response_stream(rules, messages, temperature=0.05, top_k=40, top_p=0.95):
    key = await classify_prompt(messages[-1].content)
    print(f"Classification: {classifications[key]}")

    payload = {
        "prompt": convert_to_chat_template(rules, messages),
        "temperature": temperature,
        "n_predict": -1,
        "top_k": top_k,
        "top_p": top_p,
        "key": key,
        "stream": True,
        "penalize_nl": True,
        "repeat_last_n": 0,
        "min_keep": 0
    }
    async for chunk in llama_manager.call_llama_server_stream(payload):
        yield chunk  # Stream output chunk by chunk


async def tool_selection(api_doc, user_query):
    rules = (
        "Please read the following query and select which API is required to satisfy "
        "the user request. Please do not assume anything and use only the information provided to respond.\n"
        f"Here are the available tools: {api_doc}\n"
        "Please respond as in the following JSON format:\n"
        "{\n"
        "  \"<Needed_Tool_Name_1>\": {\n"
        "    \"Params\": [\"<param_1_data>\", \"<param_2_data>\"]\n"
        "    \"Missing_Params\": [\"<missing_param_1>\", \"<missing_param_2>\"] (include this key only if there are missing parameters)\n"
        "  },\n"
        "  \"<Needed_Tool_Name_2>\": {\n"
        "    \"Params\": []\n"
        "    \"Missing_Params\": [] (include only if there are missing parameters)\n"
        "  }\n"
        "}\n"
        "If no tools are required to answer the query, return an empty JSON object: {}"
    )

    payload = {
        "prompt": convert_to_chat_template(rules, user_query),
        "temperature": 0.05,
        "n_predict": -1,
        "top_k": 40,
        "top_p": 0.9,
        "stream": False,
        "penalize_nl": True,
        "repeat_last_n": 0,
        "min_keep": 0
    }

    response = await llama_manager.call_llama_server(payload)

    # Clean and parse response content as JSON
    raw_content = response['content'].replace('\n', '').replace('<|eot_id|>', '')

    # Attempt to fix common JSON formatting issues
    fixed_content = re.sub(r'(?<=\])(?=\s*")', ',', raw_content)  # Add missing commas between objects

    try:
        # Convert response content to JSON
        json_response = json.loads(fixed_content)
    except json.JSONDecodeError:
        # Handle cases where response is not valid JSON
        json_response = {"Error": fixed_content}

    return json_response


# Classify a given prompt
def prompt_classification(prompt):
    return classify_prompt(prompt, text=True)
