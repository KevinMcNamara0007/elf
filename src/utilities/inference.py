import onnxruntime_genai as og
from fastapi import HTTPException
from src.utilities.general import CONTEXT_WINDOW, extract_text_from_pdf, extract_text_from_docx, \
    extract_text_from_excel, extract_text_from_txt, VISION_MODEL_DIR


def run_inference(prompt, image=None):
    """
    Helper function to run inference for both vision-and-text and text-only.
    :param prompt: The input prompt for inference
    :param image: Optional image for vision-and-text inference
    :return: The generated response
    """
    try:
        # Prepare model inputs
        inputs = vision_processor(prompt, images=image)

        # Set up model generator parameters
        params = og.GeneratorParams(vision_model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=CONTEXT_WINDOW)

        # Initialize the generator
        generator = og.Generator(vision_model, params)

        # Generate response tokens
        response = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            response += vision_tokenizer.decode(new_token)

        # Clean up the generator to free resources
        del generator
        return response

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during inference.")


def vision_and_text_inference(prompt, image):
    """
    Function to perform vision-and-text inference.
    :param prompt: The input prompt for inference
    :param image: Image input for the vision model
    :return: The generated response
    """
    return run_inference(prompt, image=image)


def text_inference(prompt):
    """
    Function to perform text-only inference.
    :param prompt: The input prompt for inference
    :return: The generated response
    """
    return run_inference(prompt)


async def doc_extractor(file):
    """
    Extract text from various file formats (PDF, DOCX, XLSX, TXT).
    :param file: The uploaded file object
    :return: Extracted text content in dictionary format
    """
    file_type = file.content_type
    text_content = ""

    try:
        if file_type == 'application/pdf':
            text_content = await extract_text_from_pdf(file)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text_content = await extract_text_from_docx(file)
        elif file_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            text_content = await extract_text_from_excel(file)
        elif file_type == 'text/plain':
            text_content = await extract_text_from_txt(file)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    return {"contents": text_content}


# Initialize model, processor, and tokenizer once at the module level
try:
    vision_model = og.Model(VISION_MODEL_DIR)
    vision_processor = vision_model.create_multimodal_processor()
    vision_tokenizer = vision_processor.create_stream()

except Exception as e:
    raise RuntimeError(f"Failed to load model or initialize processor: {e}")
