import os
from dotenv import load_dotenv

from omegaconf import OmegaConf

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

load_dotenv()


class LLM(object):
    def __init__(self, cfg_path):
        self.cfg = OmegaConf.load(cfg_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            **self.wrp_tokenizer_from_pretrained(**self.cfg.tokenizer.from_pretrained)
        )

        if self.cfg.model.peft:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                **self.wrp_model_from_pretrained(**self.cfg.model.from_pretrained)
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                **self.wrp_model_from_pretrained(**self.cfg.model.from_pretrained)
            )

        print(
            "Load: {}".format(
                self.cfg.model.from_pretrained.pretrained_model_name_or_path
            )
        )
        self.model.eval()

    def wrp_tokenizer_from_pretrained(self, **kwargs):
        kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs["cache_dir"] = os.environ.get("HF_CACHE_DIR", "./.cache")
        return kwargs

    def wrp_model_from_pretrained(self, **kwargs):
        kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs["cache_dir"] = os.environ.get("HF_CACHE_DIR", "./.cache")
        if "torch_dtype" in kwargs:
            kwargs["torch_dtype"] = eval(kwargs["torch_dtype"])
        return kwargs

    def get_chat(self, user_content):
        return [
            {"role": "system", "content": self.cfg.prompt.system.content},
            {"role": "user", "content": user_content},
        ]

    def __call__(self, user_content):
        if "apply_chat_template" in self.cfg.tokenizer:
            chat = self.get_chat(
                user_content,
            )
            tokenized_input = self.tokenizer.apply_chat_template(
                chat, **self.cfg.tokenizer.apply_chat_template
            ).to(self.model.device)
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

        with torch.no_grad():
            output = self.model.generate(
                tokenized_input,
                **self.cfg.model.generate,
            )[0]

        output = output[tokenized_input.size(-1) :]
        return self.tokenizer.decode(output, **self.cfg.tokenizer.decode)
