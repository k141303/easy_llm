from easy_llm import load_llm

model_name = "llm-jp/llm-jp-13b-instruct-full-ac_001_16x-dolly-ichikara_004_001_single-oasst-oasst2-v2.0"

model = load_llm(model_name)

instruction = "自然言語処理とは何か"
response = model(instruction)

print(f"### 指示:\n{instruction}")
print(f"\n\n### 応答:\n{response}")
