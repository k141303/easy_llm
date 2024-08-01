import os

from .llm import LLM, OpenAIClient, BedrockClient


def load_llm(
    model_name_or_path="llm-jp/llm-jp-13b-instruct-full-ac_001_16x-dolly-ichikara_004_001_single-oasst-oasst2-v2.0",
):
    cfg_path = os.path.join(
        os.path.dirname(__file__), "config", model_name_or_path + ".yaml"
    )
    if not os.path.exists(cfg_path):
        raise ValueError(f"Model not found: {model_name_or_path}")

    if model_name_or_path.startswith("openai"):
        return OpenAIClient(cfg_path)

    if model_name_or_path.startswith("anthropic"):
        return BedrockClient(cfg_path)
    return LLM(cfg_path)
