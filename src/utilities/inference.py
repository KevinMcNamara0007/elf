import onnxruntime_genai as og
from fastapi import HTTPException

from src.utilities.general import vision_processor, vision_model, vision_tokenizer


def vision_and_text_inference(prompt, image):
    try:
        # Prepare inputs for the model
        inputs = vision_processor(prompt, images=image)

        # Set model parameters
        params = og.GeneratorParams(vision_model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=65000)

        # Initialize the generator
        generator = og.Generator(vision_model, params)

        # Generate the response
        vision_model_response = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            vision_model_response += vision_tokenizer.decode(new_token)

        del generator
        return vision_model_response

    except Exception as e:
        print("Error during vision inference:", e)
        raise HTTPException(status_code=500, detail="An error occurred during vision inference.")


def text_inference(prompt):
    try:
        inputs = vision_processor(prompt, images=None)
        params = og.GeneratorParams(vision_model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=128000)

        generator = og.Generator(vision_model, params)

        vision_model_response = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            vision_model_response += vision_tokenizer.decode(new_token)
        del generator
        return vision_model_response
    except Exception as e:
        print("Error during text inference:", str(e))
        raise HTTPException(status_code=500, detail="An error occurred during text inference.")