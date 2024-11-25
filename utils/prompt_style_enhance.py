from modules.sdxl_styles import apply_style
from modules.util import remove_empty_str
# Fooocus GPT2 Expansion
# Algorithm created by Lvmin Zhang at 2023, Stanford
# If used inside Fooocus, any use is permitted.
# If used outside Fooocus, only non-commercial use is permitted (CC-By NC 4.0).
# This applies to the word list, vocab, model, and algorithm.


import os
import torch
import math

from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM

neg_inf = -8192.0


def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace("  ", " ")
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, "")
    return x


class FooocusExpansion:
    def __init__(
        self,
        positive_txt_path,
        repo_id="metercai/SimpleSDXL",
        sub_folder="prompt_expansion/fooocus_expansion",
        load_device="cpu",
        use_fp16=True,
    ):
        self.load_device = load_device

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=sub_folder)

        positive_words = open(positive_txt_path, encoding="utf-8").read().splitlines()
        positive_words = ["Ġ" + x.lower() for x in positive_words if x != ""]

        self.logits_bias = (
            torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf
        )

        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])

        print(f"Fooocus V2 Expansion: Vocab with {len(debug_list)} words.")

        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id, subfolder=sub_folder
        ).to(load_device)
        self.model.eval()

        if use_fp16:
            self.model.half()
        print(
            f"Fooocus Expansion engine loaded for {load_device}, use_fp16 = {use_fp16}."
        )

    @torch.no_grad()
    @torch.inference_mode()
    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(scores)

        bias = self.logits_bias.clone()
        bias[0, input_ids[0].to(bias.device).long()] = neg_inf
        bias[0, 11] = 0

        return scores + bias

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, prompt):
        if prompt == "":
            return ""

        prompt = safe_str(prompt) + ","

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(
            self.load_device
        )
        tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data[
            "attention_mask"
        ].to(self.load_device)

        current_token_length = int(tokenized_kwargs.data["input_ids"].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        if max_new_tokens == 0:
            return prompt[:-1]

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(
            **tokenized_kwargs,
            top_k=100,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            logits_processor=LogitsProcessorList([self.logits_processor]),
        )

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = safe_str(response[0])

        return result


def enhance_prompt(prompt, 
                   negative_prompt="", 
                   use_fooocus_expansion=True, 
                   positive_txt_path = "",
                   device = "cpu",
                   style_selections = ['Fooocus Enhance', 'Fooocus Sharp']
):
    positive_basic_workloads = []
    negative_basic_workloads = []
    for s in style_selections:
        p, n = apply_style(s, positive=prompt)
        positive_basic_workloads = positive_basic_workloads + p
        negative_basic_workloads = negative_basic_workloads + n
    negative_basic_workloads.append(negative_prompt)

    positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=prompt)[0]
    negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=negative_prompt)[0]

    if use_fooocus_expansion:
        assert positive_txt_path != "", "Positive txt path is required when using Fooocus Expansion"
        final_expansion = FooocusExpansion(positive_txt_path="./positive.txt",
                            #   repo_id="metercai/SimpleSDXL",
                            #   sub_folder="prompt_expansion/fooocus_expansion",
                            load_device=device,
                            use_fp16=True)
        positive_basic_workloads = final_expansion(positive_basic_workloads)
        negative_basic_workloads = final_expansion(negative_basic_workloads)
    positive_basic_workloads = prompt + "," + positive_basic_workloads if prompt != "" else positive_basic_workloads
    negative_basic_workloads = negative_prompt + "," + negative_basic_workloads if negative_prompt != "" else negative_basic_workloads
    return positive_basic_workloads, negative_basic_workloads
