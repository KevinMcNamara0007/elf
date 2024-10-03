import numpy as np
from fastapi import HTTPException
from src.utilities.general import classifier_manager, classifications


async def classify_prompt(prompt, max_len=100, text=False):
    classifier = classifier_manager.classifier
    tokenizer = classifier_manager.tokenizer

    try:

        # Tokenize prompt
        input_tokens = tokenizer.texts_to_sequences([prompt])

        # Pad input sequence
        max_seq_length = max_len
        input_tokens_padded = np.zeros((1, max_seq_length), dtype=np.float32)
        input_tokens_padded[:, :len(input_tokens[0])] = input_tokens[0][:max_seq_length]

        # Perform inference
        inputs = {classifier.get_inputs()[0].name: input_tokens_padded}
        prediction = classifier.run(None, inputs)[0]

        # Process prediction
        binary_prediction = [0 if x < 0.5 else 1 for x in prediction[0]]

        # Check for ambiguity and return result
        if binary_prediction.count(1) != 1:
            return 1

        # Return the index or the text result
        if not text:
            return binary_prediction.index(1)
        return classifications[binary_prediction.index(1)]

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
