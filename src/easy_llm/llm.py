import os

import requests

from dotenv import load_dotenv

from omegaconf import OmegaConf

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    pipeline,
)
from peft import AutoPeftModelForCausalLM

from openai import AzureOpenAI, BadRequestError

import boto3

from vllm import LLM as VLLM
from vllm import SamplingParams

load_dotenv(override=True)


class LLM(object):
    def __init__(self, cfg_path, max_new_tokens=None):
        self.cfg = OmegaConf.load(cfg_path)

        if max_new_tokens is not None:
            self.cfg.max_new_tokens = max_new_tokens

        self.tokenizer = None
        if "tokenizer" in self.cfg:
            self.tokenizer = AutoTokenizer.from_pretrained(
                **self.wrp_tokenizer_from_pretrained(
                    **self.cfg.tokenizer.from_pretrained
                )
            )

        self.streamer = None
        if "textstreamer" in self.cfg:
            self.streamer = TextStreamer(
                self.tokenizer,
                **self.cfg.textstreamer,
            )

        self.model = None
        if "model" in self.cfg:
            if self.cfg.model.peft:
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    **self.wrp_model_from_pretrained(**self.cfg.model.from_pretrained)
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    **self.wrp_model_from_pretrained(**self.cfg.model.from_pretrained)
                )

        self.pipeline = None
        if "pipeline" in self.cfg:
            self.pipeline = pipeline(
                **self.wrp_pipeline_init(**self.cfg.pipeline.init),
            )

        self.vllm = None
        if "vllm" in self.cfg:
            self.vllm = VLLM(
                **self.wrp_vllm_init(**self.cfg.vllm.init),
            )

        self.sample_params = None
        if "sample_params" in self.cfg:
            self.sample_params = SamplingParams(**self.cfg.sample_params)

        if self.model is not None and self.cfg.model.get("eval"):
            self.model.eval()

        print(f"Load: {self.cfg.name_or_path}")

    def wrp_tokenizer_from_pretrained(self, **kwargs):
        kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs["cache_dir"] = os.environ.get("HF_CACHE_DIR", "./.cache")
        return kwargs

    def wrp_model_from_pretrained(self, **kwargs):
        kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs["cache_dir"] = os.environ.get("HF_CACHE_DIR", "./.cache")
        if "torch_dtype" in kwargs and kwargs["torch_dtype"] != "auto":
            kwargs["torch_dtype"] = eval(kwargs["torch_dtype"])
        return kwargs

    def wrp_pipeline_init(self, **kwargs):
        if self.model is not None:
            kwargs["model"] = self.model
        if self.tokenizer is not None:
            kwargs["tokenizer"] = self.tokenizer
        kwargs["model_kwargs"] = {
            "cache_dir": os.environ.get("HF_CACHE_DIR", "./.cache"),
        }
        if "torch_dtype" in kwargs and kwargs["torch_dtype"] != "auto":
            kwargs["torch_dtype"] = eval(kwargs["torch_dtype"])
        return kwargs

    def wrp_vllm_init(self, **kwargs):
        kwargs["download_dir"] = os.environ.get("HF_CACHE_DIR", "./.cache")
        return kwargs

    def wrp_model_generate(self, **kwargs):
        if type(kwargs.get("pad_token_id")) is str:
            kwargs["pad_token_id"] = eval(kwargs["pad_token_id"])
        if type(kwargs.get("eos_token_id")) is str:
            kwargs["eos_token_id"] = eval(kwargs["eos_token_id"])
        if type(kwargs.get("bos_token_id")) is str:
            kwargs["bos_token_id"] = eval(kwargs["bos_token_id"])
        return kwargs

    def wrp_vllm_generate(self, **kwargs):
        if self.sample_params is not None:
            kwargs["sampling_params"] = self.sample_params
        return kwargs

    def get_messages(self, model_input):
        messages = []
        if self.cfg.get("prompt") is not None:
            messages.append(
                {"role": "system", "content": self.cfg.prompt.system.content}
            )
        if isinstance(model_input, str):
            messages.append({"role": "user", "content": model_input})
        elif isinstance(model_input, list):
            for i, content in enumerate(model_input):
                if i % 2 == 0:
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "assistant", "content": content})
        else:
            raise ValueError("Input must be str or list")
        return messages

    def get_prompt(self, model_input):
        if isinstance(model_input, str):
            prompt = self.cfg.prompt.format(model_input=model_input)
        elif isinstance(model_input, list):
            raise NotImplementedError("Prompt is not implemented for multi-turn input")
        else:
            raise ValueError("Input must be str or list")
        return prompt

    def pipeline_inference(self, model_input):
        messages = self.get_messages(model_input)

        if "apply_chat_template" in self.cfg.get("tokenizer", {}):
            messages = self.tokenizer.apply_chat_template(
                messages, **self.cfg.tokenizer.apply_chat_template
            )

        if self.cfg.pipeline.init.task == "text-generation":
            messages = self.pipeline(messages, **self.cfg.pipeline.call)[-1][
                "generated_text"
            ]
            if type(messages) is list:
                messages = messages[-1]["content"]
            return messages
        return self.pipeline(messages, **self.cfg.pipeline.call)[-1]["content"]

    def regacy_inference(self, model_input):
        if "apply_chat_template" in self.cfg.tokenizer:
            messages = self.get_messages(
                model_input,
            )
            chat_template = self.tokenizer.apply_chat_template(
                messages, **self.cfg.tokenizer.apply_chat_template
            )
            if not self.cfg.tokenizer.apply_chat_template.get("tokenize", True):
                tokenized_input = self.tokenizer.encode(
                    chat_template, **self.cfg.tokenizer.encode
                ).to(self.model.device)
            else:
                tokenized_input = chat_template.to(self.model.device)
        else:
            prompt = self.get_prompt(model_input)
            if "encode" in self.cfg.tokenizer:
                tokenized_input = self.tokenizer.encode(
                    prompt, **self.cfg.tokenizer.encode
                ).to(self.model.device)
            else:
                tokenized_input = self.tokenizer(
                    prompt, **self.cfg.tokenizer.call
                ).input_ids.to(self.model.device)

        with torch.no_grad():  # Regacy Inference Mode
            if self.streamer is not None:
                output = self.model.generate(
                    tokenized_input,
                    streamer=self.streamer,
                    **self.wrp_model_generate(**self.cfg.model.generate),
                )[0]
            else:
                output = self.model.generate(
                    tokenized_input,
                    **self.wrp_model_generate(**self.cfg.model.generate),
                )[0]

        output = output[tokenized_input.size(-1) :]
        return self.tokenizer.decode(output, **self.cfg.tokenizer.decode)

    def vllm_inference(self, model_input):
        if "apply_chat_template" in self.cfg.tokenizer:
            messages = self.get_messages(
                model_input,
            )
            chat_template = self.tokenizer.apply_chat_template(
                messages, **self.cfg.tokenizer.apply_chat_template
            )
        else:
            raise NotImplementedError(
                "VLLM Inference Mode is not implemented without apply_chat_template"
            )

        output = self.vllm.generate(
            chat_template,
            **self.wrp_vllm_generate(**self.cfg.vllm.generate),
        )
        return output[0].outputs[0].text

    def __call__(self, model_input, max_new_tokens=None):
        if max_new_tokens is not None:
            self.cfg.max_new_tokens = max_new_tokens

        if self.pipeline is not None:
            return self.pipeline_inference(model_input)

        if self.vllm is not None:
            return self.vllm_inference(model_input)

        return self.regacy_inference(model_input)


class OpenAIClient(LLM):
    def __init__(self, cfg_path, max_new_tokens=None):
        self.cfg = OmegaConf.load(cfg_path)

        if max_new_tokens is not None:
            self.cfg.max_new_tokens = max_new_tokens

        self.client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version="2023-05-15",
            azure_endpoint=os.environ["OPENAI_API_BASE"],
        )

    def __call__(self, model_input, max_new_tokens=None):
        if max_new_tokens is not None:
            self.cfg.max_new_tokens = max_new_tokens

        messages = self.get_messages(model_input)

        response = self.client.chat.completions.create(
            messages=messages, **self.cfg.client.chat.completions.create
        )

        return response.choices[0].message.content


class BedrockClient(object):
    def __init__(self, cfg_path):
        self.cfg = OmegaConf.load(cfg_path)

        self.client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )

        self.system_prompts = [
            {
                "text": self.cfg.prompt.system.content,
            }
        ]

    def get_messages(self, model_input):
        messages = []
        if isinstance(model_input, str):
            messages.append({"role": "user", "content": [{"text": model_input}]})
        elif isinstance(model_input, list):
            for i, content in enumerate(model_input):
                if i % 2 == 0:
                    messages.append({"role": "user", "content": [{"text": content}]})
                else:
                    messages.append(
                        {"role": "assistant", "content": [{"text": content}]}
                    )
        else:
            raise ValueError("Input must be str or list")
        return messages

    def __call__(self, model_input, max_new_tokens=None):
        if max_new_tokens is not None:
            self.cfg.max_new_tokens = max_new_tokens

        messages = self.get_messages(model_input)

        inferenceConfig = dict(self.cfg.inferenceConfig)
        response = self.client.converse(
            modelId=self.cfg.name,
            messages=messages,
            inferenceConfig=inferenceConfig,
            system=self.system_prompts,
        )

        return response["output"]["message"]["content"][0]["text"]
