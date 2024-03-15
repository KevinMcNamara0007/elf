from fastapi import HTTPException
from src.utilities.general import expert_models


async def load_model(category):
    try:
        return expert_models.get(category)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model: {exc}")


async def categorize(prompt):
    try:
        expert = await load_model("General")
        classification = expert.create_chat_completion(
            messages=[
                {
                    "role": "assistant",
                    "content": f"You are a classification expert. "
                               f"Respond with the category label."
                               f"DO NOT ANSWER THE PROMPT."
                               f"DO NOT EXPLAIN."
                               f"Classify the following prompt into one of the following knowledge categories: "
                               f"{list(expert_models.keys())}."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=.05
        )["choices"][0]["message"]["content"]
        for category in expert_models.keys():
            if category in classification:
                return category
        return "General"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Coult not categorize prompt: {exc}")


async def fetch_expert_response(prompt, temperature, category):
    try:
        expert = await load_model(category)
        return expert.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch response from model: {exc}")
