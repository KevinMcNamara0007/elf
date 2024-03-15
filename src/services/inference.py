from src.utilities.general import expert_models
from src.utilities.inference import categorize, fetch_expert_response


async def get_all_models():
    return [f"{category} expert -> {model_path.split('efs/models/')[1].replace('.gguf', '')}" for category, model_path
            in expert_models.items()]


async def get_expert_response(prompt, temperature=.05):
    category = await categorize(prompt)
    return await fetch_expert_response(prompt, temperature, category)
