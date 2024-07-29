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

### シングルターン

```bash
python3 example/single_turn_inference.py
```

### マルチターン

```python
python3 example/multi_turn_inference.py
```

## 対応済みモデル

> [!NOTE]
> [apply_chat_template()](https://huggingface.co/docs/transformers/main/ja/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)や[pipeline()](https://huggingface.co/docs/transformers/ja/main_classes/pipelines#transformers.pipeline)を使用しないモデル(※1)はマルチターン非対応です。

### 国産
- cyberagent/calm2-7b-chat ※1
- cyberagent/calm3-22b-chat
- elyza/Llama-3-ELYZA-JP-8B
- Fugaku-LLM/Fugaku-LLM-13B-instruct ※1
- lightblue/ao-karasu-72B
- llm-jp/llm-jp-13b-instruct-full-dolly-ichikara_004_001_single-oasst-oasst2-v2.0
- llm-jp/llm-jp-13b-instruct-full-ac_001_16x-dolly-ichikara_004_001_single-oasst-oasst2-v2.0
- matsuo-lab/weblab-10b-instruction-sft ※1
- pfnet/plamo-13b-instruct ※1
- rinna/nekomata-14b-instruction ※1
- stockmark/stockmark-100b-instruct-v0.1 ※1
- tokyotech-llm/Swallow-70b-instruct-hf ※1
- tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1
- tokyotech-llm/Llama-3-Swallow-70B-Instruct-v0.1

### 海外産
- microsoft/Phi-3-medium-4k-instruct
- mistralai/Mistral-7B-Instruct-v0.3
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3-70B-Instruct
- google/gemma-1.1-7b-it
- Qwen/Qwen2-72B-Instruct

## 現在非対応

- karakuri-ai/karakuri-lm-70b-chat-v0.1
(transformers>=4.42でConversationクラスが削除されたため動きません。)