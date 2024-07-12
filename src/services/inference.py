from src.utilities.general import classifications, CONTEXT_WINDOW
from src.utilities.inference import classify_prompt, fetch_llama_cpp_response


async def get_all_models():
    return [f"{classification['Category']} expert -> {classification['Model'].metadata['general.name']}" for
            classification
            in classifications.values()]


async def get_expert_response(rules, messages, temperature=.05, max_tokens=CONTEXT_WINDOW):
    # key = await classify_prompt(messages[-1]["content"])
    key = 1
    # Fetch response
    cont_response = await fetch_llama_cpp_response(rules, messages, temperature, key, max_tokens)
    # Extract model response
    final_response = cont_response['content']
    # Extract other useful data from original response
    p_tokens = cont_response['tokens_evaluated']
    c_tokens = cont_response['tokens_predicted']
    # Extract finish reason from response
    finish_reason = cont_response['truncated']
    # If the finish_reason was due to length, maintain a loop to generate the rest of the answer.
    upper_bound_token_limit = (max_tokens - (c_tokens + p_tokens))
    while finish_reason and upper_bound_token_limit > 0:
        continuation_prompt = [
            {'role': 'User', 'content': f"Please continue the response. ORIGINAL PROMPT: {messages[-1]['content']}"},
            {'role': 'assistant', 'content': final_response}
        ]

        cont_response = await fetch_llama_cpp_response(continuation_prompt, temperature, key, upper_bound_token_limit)
        finish_reason = cont_response['truncated']
        final_response += " " + cont_response['content']

        p_tokens += cont_response['tokens_evaluated']
        c_tokens += cont_response['tokens_predicted']
        upper_bound_token_limit -= (c_tokens + p_tokens)
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
    return await classify_prompt(prompt)

