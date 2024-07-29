from easy_llm import load_llm

model_name = "llm-jp/llm-jp-13b-instruct-full-ac_001_16x-dolly-ichikara_004_001_single-oasst-oasst2-v2.0"

instruction1 = "自然言語処理とは何か"

model = load_llm(model_name)
response1 = model([instruction1])

instruction2 = "もっと簡潔に教えてください。"

response2 = model([instruction1, response1, instruction2])

print(f"### 指示:\n{instruction1}")
print(f"\n\n### 応答:\n{response1}")
print(f"\n\n### 指示:\n{instruction2}")
print(f"\n\n### 応答:\n{response2}")
