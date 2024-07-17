import math

from src.utilities.general import classifications, CONTEXT_WINDOW, INPUT_WINDOW
from src.utilities.inference import fetch_llama_cpp_response, classify_prompt


async def get_all_models():
    return [f"{classification['Category']} expert -> {classification['Model'].metadata['general.name']}" for
            classification
            in classifications.values()]


async def get_expert_response(messages, temperature=.05, max_tokens=int(CONTEXT_WINDOW)-int(INPUT_WINDOW)):
    key = await classify_prompt(messages[-1]["content"])
    print("Classification: {}".format(classifications.get(key)["Category"]))
    rough_token_count = int(math.ceil(len(str(messages)) / 5))
    # Fetch response
    cont_response = await fetch_llama_cpp_response(messages, temperature, key, rough_token_count)
    # Extract model response
    final_response = cont_response['content']
    # Extract other useful data from original response
    p_tokens = cont_response['tokens_evaluated']
    c_tokens = cont_response['tokens_predicted']
    # Extract finish reason from response
    finish_reason = cont_response['truncated']
    # If the finish_reason was due to length, maintain a loop to generate the rest of the answer.
    upper_bound_token_limit = INPUT_WINDOW
    while finish_reason and upper_bound_token_limit > 0:
        continuation_prompt = [
            {'role': 'User', 'content': f"Please continue the response. ORIGINAL PROMPT: {messages[-1]['content']}"},
            {'role': 'assistant', 'content': final_response}
        ]
        cont_response = await fetch_llama_cpp_response(continuation_prompt, temperature, key, c_tokens)
        finish_reason = cont_response['truncated']
        final_response += " " + cont_response['content']

        p_tokens += cont_response['tokens_evaluated']
        c_tokens += cont_response['tokens_predicted']
        upper_bound_token_limit = INPUT_WINDOW - c_tokens
    return {
        'usage': {
            'total_tokens': p_tokens + c_tokens,
            'prompt_tokens': p_tokens,
            'completion_tokens': c_tokens,
        },
        'choices': [{
            'message': {
                'content': final_response
            },
            'finish_reason': 'stop'
        }],
        'timings': cont_response['timings']
    }


async def prompt_classification(prompt):
    return await classify_prompt(prompt, text=True)

