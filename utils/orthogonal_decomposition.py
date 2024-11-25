import re
import torch

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def orthogonal_decomposition(raw_emb: torch.Tensor, remove_emb: torch.Tensor) -> torch.Tensor:
    projected_vector_magnitude = raw_emb.dot(remove_emb) / remove_emb.norm()
    projected_vector = projected_vector_magnitude * remove_emb / remove_emb.norm()
    return raw_emb - projected_vector


def sks_decompose(
    prompt: str,
    prompt_emb: torch.Tensor,
    to_decopose_embeds: torch.Tensor,
    decompose_words_num: int,
    prefix_prompt:str = "",
) -> torch.Tensor:

    prompt = normalize_spaces(prompt.lower().strip())
    prompt_words = prompt.split(" ")

    prefix_prompt = normalize_spaces(prefix_prompt.lower().strip())

    if prefix_prompt == "":
        prefix_prompt_len = 0 + 1
    else:
        prefix_prompt_len = len(prefix_prompt.split(" ")) + 1

    # get index of "sks"
    for i in range(len(prompt_words)):
        if prompt_words[i] == "sks":
            ind_sks = i + 1 
            break
    else:
        raise ValueError(f"Prompt {prompt} does not contain 'sks'")

    # # get index of remove_words
    inds_replace = []
    # for word in remove_words:
    #     word = word.lower()
    #     for i in range(len(prompt_words)):
    #         if prompt_words[i] == word:
    #             inds_replace.append(i + 1)
    #             break

    # for ind_replace in inds_replace:
    #     prompt_emb[ind_sks, ...] = orthogonal_decomposition(
    #         prompt_emb[ind_sks, ...], raw_prompt_embeds[ind_replace, ...]
    #     )

    for ind_de in range(prefix_prompt_len, decompose_words_num + prefix_prompt_len):
        # for i in range(decompose_words_num):
        for ind in range(1, len(prompt_words) + 1):
            prompt_emb[ind, ...] = orthogonal_decomposition(
                prompt_emb[ind, ...], to_decopose_embeds[ind_de, ...]
            )

    return prompt_emb
