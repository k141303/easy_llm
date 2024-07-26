import os
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

from vllm import LLM as VLLM
from vllm import SamplingParams

load_dotenv()


class LLM(object):
    def __init__(self, cfg_path):
        self.cfg = OmegaConf.load(cfg_path)

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
            kwargs["sample_params"] = self.sample_params
        return kwargs

    def get_chat(self, user_content):
        chat = []
        if self.cfg.get("prompt") is not None:
            chat.append({"role": "system", "content": self.cfg.prompt.system.content})
        chat.append({"role": "user", "content": user_content})
        return chat

    def pipeline_inference(self, user_content):
        messages = self.get_chat(user_content)

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

    def regacy_inference(self, user_content):
        if "apply_chat_template" in self.cfg.tokenizer:
            chat = self.get_chat(
                user_content,
            )
            chat_template = self.tokenizer.apply_chat_template(
                chat, **self.cfg.tokenizer.apply_chat_template
            )
            if not self.cfg.tokenizer.apply_chat_template.get("tokenize", True):
                tokenized_input = self.tokenizer.encode(
                    chat_template, **self.cfg.tokenizer.encode
                ).to(self.model.device)
            else:
                tokenized_input = chat_template.to(self.model.device)
        else:
            prompt = self.cfg.prompt.format(user_content=user_content)
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

    def vllm_inference(self, user_content):
        if "apply_chat_template" in self.cfg.tokenizer:
            chat = self.get_chat(
                user_content,
            )
            chat_template = self.tokenizer.apply_chat_template(
                chat, **self.cfg.tokenizer.apply_chat_template
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

    def __call__(self, user_content):
        if self.pipeline is not None:
            return self.pipeline_inference(user_content)

        if self.vllm is not None:
            return self.vllm_inference(user_content)

        return self.regacy_inference(user_content)
