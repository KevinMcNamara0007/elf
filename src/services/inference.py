from src.utilities.general import classifications
from src.utilities.inference import fetch_llama_cpp_response, classify_prompt, fetch_llama_cpp_response_stream, fetch_pro


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
