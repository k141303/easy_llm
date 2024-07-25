Hugging Face上のモデルを各公式設定で簡単に動かすためのライブラリ

## インストール

```bash
pip install .
```

## 環境変数

```bash
export HF_TOKEN=YOUR_HUGGINGFACE_ACCESS_TOKEN
export HF_CACHE_DIR=DIR_NAME
```

## 使い方

```python
from easy_llm import load_llm

MODEL_NAME = "llm-jp/llm-jp-13b-instruct-full-ac_001_16x-dolly-ichikara_004_001_single-oasst-oasst2-v2.0"

model = load_llm(MODEL_NAME)
print(model("自然言語処理とは何か"))
```

## 対応済みモデル

### 国産
- cyberagent/calm2-7b-chat
- cyberagent/calm3-22b-chat
- Fugaku-LLM/Fugaku-LLM-13B-instruct
- karakuri-ai/karakuri-lm-70b-chat-v0.1
- lightblue/ao-karasu-72B
- llm-jp/llm-jp-13b-instruct-full-dolly-ichikara_004_001_single-oasst-oasst2-v2.0
- llm-jp/llm-jp-13b-instruct-full-ac_001_16x-dolly-ichikara_004_001_single-oasst-oasst2-v2.0
- matsuo-lab/weblab-10b-instruction-sft
- pfnet/plamo-13b-instruct
- rinna/nekomata-14b-instruction
- stockmark/stockmark-100b-instruct-v0.1
- tokyotech-llm/Swallow-70b-instruct-hf

### 海外産
- microsoft/Phi-3-medium-4k-instruct
- mistralai/Mistral-7B-Instruct-v0.3
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3-70B-Instruct
- google/gemma-1.1-7b-it
- Qwen/Qwen2-72B-Instruct
