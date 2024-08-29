from src.utilities.inference import text_inference


async def vision_text_inference(prompt):
    formatted_prompt = f"<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"
    return text_inference(formatted_prompt)