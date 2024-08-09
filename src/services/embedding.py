import numpy as np
from fastapi import HTTPException

from src.utilities.general import embedding_tokenizer, embedding_model


async def create_embedding(input_text):
    try:
        model_inputs = embedding_tokenizer(input_text, return_tensors='np')

        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        token_type_ids = model_inputs.get('token_type_ids', np.zeros_like(input_ids))

        input_ids = input_ids.astype(np.int64)
        attention_mask = attention_mask.astype(np.int64)
        token_type_ids = token_type_ids.astype(np.int64)

        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if 'token_type_ids' in model_inputs:
            onnx_inputs['token_type_ids'] = token_type_ids

        onnx_outputs = embedding_model.run(None, onnx_inputs)
        return onnx_outputs[0]
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not generate embedding: {str(e)}")
