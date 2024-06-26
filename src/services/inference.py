from src.utilities.general import classifications, context_window
from src.utilities.inference import fetch_expert_response, classify_prompt


async def get_all_models():
    return [f"{classification['Category']} expert -> {classification['Model'].metadata['general.name']}" for
            classification
            in classifications.values()]


async def get_expert_response(messages, temperature=.05, max_tokens=context_window):
    key = await classify_prompt(messages[-1]["content"])
    # Fetch response
    expert_response = await fetch_expert_response(messages, temperature, key, context_window)
    # Extract model response
    final_response = expert_response['choices'][0]['message']['content']
    # Extract other useful data from original response
    p_tokens = expert_response['usage']['prompt_tokens']
    c_tokens = expert_response['usage']['completion_tokens']
    # Extract finish reason from response
    finish_reason = expert_response['choices'][0]['finish_reason']
    # If the finish_reason was due to length, maintain a loop to generate the rest of the answer.
    while 'length' == finish_reason:
        print(finish_reason)
        print(final_response)
        continuation_prompt = [
            {'role': 'user', 'content': f"Please continue the response. ORIGINAL PROMPT: {messages[-1]['content']}"},
            {'role': 'assistant', 'content': final_response}
        ]

        cont_response = await fetch_expert_response(continuation_prompt, temperature, key, context_window)
        finish_reason = cont_response['choices'][0]['finish_reason']
        if finish_reason == 'exceed':
            break
        final_response += " " + cont_response['choices'][0]['message']['content']

        p_tokens += cont_response['usage']['prompt_tokens']
        c_tokens += cont_response['usage']['completion_tokens']

    expert_response['usage']['total_tokens'] = p_tokens + c_tokens
    expert_response['usage']['prompt_tokens'] = p_tokens
    expert_response['usage']['completion_tokens'] = c_tokens
    expert_response['choices'][0]['finish_reason'] = finish_reason
    expert_response['choices'][0]['message']['content'] = final_response
    return expert_response


async def prompt_classification(prompt):
    return await classify_prompt(prompt)

