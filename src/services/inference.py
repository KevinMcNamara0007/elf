import math
from src.utilities.general import classifications, INPUT_WINDOW
from src.utilities.inference import fetch_llama_cpp_response, classify_prompt


async def get_expert_response(rules, messages, temperature=.05, top_k=40, top_p=0.95):
    """
    Fetches response from llama-server and recalls if generation is truncated. Returns the full response.
    """
    key = await classify_prompt(messages[-1].content)
    print("Classification: {}".format(classifications.get(key)["Category"]))
    rough_token_count = int(math.ceil(len(str(messages)) / 5))
    # Fetch response
    cont_response = await fetch_llama_cpp_response(rules, messages, temperature, key, rough_token_count, top_k, top_p)
    # Extract model response
    final_response = cont_response['content']
    # Extract other useful data from original response
    p_tokens = cont_response['tokens_evaluated']
    c_tokens = cont_response['tokens_predicted']
    # Extract finish reason from response
    finish_reason = cont_response['truncated']
    # If the finish_reason was due to length, maintain a loop to generate the rest of the answer.
    upper_bound_token_limit = INPUT_WINDOW
    continuation_rules = f"Please continue the response. ORIGINAL PROMPT: {messages[-1]['content']}"
    while finish_reason and upper_bound_token_limit > 0:
        continuation_prompt = [
            {'role': 'assistant', 'content': final_response}
        ]
        cont_response = await fetch_llama_cpp_response(
            rules=continuation_rules,
            messages=continuation_prompt,
            temperature=temperature,
            key=key,
            input_tokens=c_tokens,
            top_p=top_p,
            top_k=top_k,
        )
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


def prompt_classification(prompt):
    """
    Classifies a given prompt using the cnn classifier.
    """
    return classify_prompt(prompt, text=True)

