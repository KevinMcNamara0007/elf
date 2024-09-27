import json

from fastapi import HTTPException

from src.utilities.general import classifications
from src.utilities.inference import fetch_llama_cpp_response, classify_prompt, fetch_llama_cpp_response_stream, \
    fetch_pro, call_openai, call_claude, fetch_pro_stream


async def get_expert_response(rules, messages, temperature=0.05, top_k=40, top_p=.95):
    """
    Fetches response from llama-server and recalls if generation is truncated. Returns the full response.
    """
    key = await classify_prompt(messages[-1].content)
    print("Classification: {}".format(classifications.get(key)["Category"]))
    response = await fetch_llama_cpp_response(
        rules=rules,
        messages=messages,
        temperature=temperature,
        key=key,
        top_p=top_p,
        top_k=top_k,
    )
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


async def get_pro_response(prompt, output_tokens):
    """
    Fetches response from llama-server and recalls if generation is truncated. Returns the full response.
    """
    key = await classify_prompt(prompt)
    print("Classification: {}".format(classifications.get(key)["Category"]))
    response = await fetch_pro(prompt=prompt,output_tokens=output_tokens, key=key)
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


async def get_pro_response_stream(prompt, output_tokens):
    """
    Fetches response from llama-server and recalls if generation is truncated. Returns the full response.
    """
    key = await classify_prompt(prompt)
    async for chunk in fetch_pro_stream(prompt=prompt,output_tokens=output_tokens, key=key):
        arr = chunk.split(': ', 1)[1]
        data_dict = json.loads(arr)
        content = data_dict.get('content')
        yield content


async def get_expert_response_stream(rules, messages, temperature=0.05, top_k=40, top_p=0.95):
    """
    Fetches response from llama-server and streams the result. Yields each chunk as it's received.
    """
    key = await classify_prompt(messages[-1].content)
    print(f"Classification: {classifications.get(key)['Category']}")

    # Call the async generator function and stream its output
    async for chunk in fetch_llama_cpp_response_stream(
            rules=rules,
            messages=messages,
            temperature=temperature,
            key=key,
            top_p=top_p,
            top_k=top_k,
    ):
        yield chunk  # Yield each chunk as it is received


def prompt_classification(prompt):
    """
    Classifies a given prompt using the cnn classifier.
    """
    return classify_prompt(prompt, text=True)


async def route_llm_request(prompt: str, llm_name: str, api_key: str):
    """
    Routes the request to the appropriate LLM service based on the llm_name provided by the user.

    :param prompt: The prompt for the LLM.
    :param llm_name: The name of the LLM service (e.g., "openai", "claude").
    :param api_key: The API key for the selected LLM service.
    :return: The response from the selected LLM.
    """
    if llm_name.lower() == 'openai':
        return await call_openai(prompt, api_key)
    elif llm_name.lower() == 'claude':
        return await call_claude(prompt, api_key)
    else:
        raise HTTPException(status_code=400, detail="Unsupported LLM provider")
