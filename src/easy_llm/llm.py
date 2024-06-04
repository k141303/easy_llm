import os
from dotenv import load_dotenv

from omegaconf import OmegaConf

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

load_dotenv()


class LLM(object):
    def __init__(self, cfg_path):
        self.cfg = OmegaConf.load(CONFIG_PATH)

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
        self.model.eval()

    def wrp_tokenizer_from_pretrained(self, **kwargs):
        kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs["cache_dir"] = "./.cache"
        return kwargs

    def wrp_model_from_pretrained(self, **kwargs):
        kwargs["token"] = os.environ["HF_TOKEN"]
        kwargs["cache_dir"] = "./.cache"
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
            tokenized_input = self.tokenizer.encode(
                prompt, **self.cfg.tokenizer.encode
            ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                tokenized_input,
                **self.cfg.model.generate,
            )[0]

        output = output[tokenized_input.size(-1) :]
        return self.tokenizer.decode(output, **self.cfg.tokenizer.decode)


CONFIG_PATH = "config/model/Fugaku-LLM/Fugaku-LLM-13B-instruct.yaml"


def main():
    model = LLM(CONFIG_PATH)
    print(model("自然言語処理とは何か"))


if __name__ == "__main__":
    main()
