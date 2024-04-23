import logging

from src.utilities.general import classifications, pipe
from src.utilities.inference import fetch_expert_response, classify_prompt


async def get_all_models():
    return [f"{classification['Category']} expert -> {classification['Model'].metadata['general.name']}" for
            classification
            in classifications.values()]


async def get_expert_response(messages, temperature=.05):
    key = await classify_prompt(messages[-1]["content"])
    return await fetch_expert_response(messages, temperature, key)


async def prompt_classification(prompt):
    return await classify_prompt(prompt)

