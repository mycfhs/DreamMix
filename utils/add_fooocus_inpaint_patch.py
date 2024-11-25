import re
import torch
from tqdm import tqdm

def add_fooocus_inpaint_patch(
    unet,
    model_path,
) -> None:
    
    fooocus_model_dict = load_patch_model_dict(model_path, unet.device)
    fooocus_model_dict_remap = process_fooocus_dict(fooocus_model_dict, unet.config)
    unet_state_dict = unet.state_dict()
    print("Loading fooocus params to unet...")
    for key in tqdm(fooocus_model_dict_remap.keys()):
        if key not in unet.state_dict().keys():
            raise ValueError(f"not find key :{key} !")
        unet_state_dict[key] += fooocus_model_dict_remap[key]
    unet.load_state_dict(unet_state_dict)

    print("Fooocus model loaded!")


def load_patch_model_dict(model_path, device):
    fooocus_model_dict = torch.load(
        model_path,
        map_location=device,
    )
    return fooocus_model_dict


def process_fooocus_dict(fooocus_model_dict, unet_config):
    # print("Processing fooocus model dict...")
    fooocus_model_dict, other_dict = dict_process_0(fooocus_model_dict)
    fooocus_model_dict = dict_process_1(fooocus_model_dict.copy(), unet_config)
    fooocus_model_dict = dict_process_2(fooocus_model_dict.copy())
    return {**fooocus_model_dict, **other_dict}


def dict_process_0(fooocus_model_dict):
    new_fooocus_dict = {}
    other_dict = {}
    alpha = 1.0
    sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]

    for k in tqdm(fooocus_model_dict.keys()):
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            final_key = k.split(".")[-1]
            key_lora = k[len("diffusion_model.") : -len(final_key)-1].replace(".", "_")
            w1, w_min, w_max = fooocus_model_dict[k]
            w1 = w1.to(dtype=torch.float32)
            w_min = w_min.to(dtype=torch.float32)
            w_max = w_max.to(dtype=torch.float32)
            w1 = (w1 / 255.0) * (w_max - w_min) + w_min  # fooocus 的计算方法
            fooocus_weight = alpha * w1
            fooocus_weight = fooocus_weight.to(dtype=torch.float16)
            if any(p in key_lora for p in sgm_patterns) and len(key_lora.split("_")) != 4:
                new_fooocus_dict["lora_unet_{}.{}".format(key_lora, final_key)] = fooocus_weight
            else:
                other_key = replace_other_key("{}.{}".format(key_lora, final_key))
                other_dict[other_key] = fooocus_weight
    return new_fooocus_dict, other_dict

def replace_other_key(key):
    key = key.replace("input_blocks_0_0", "conv_in")
    key = key.replace("middle_block_1_norm", "mid_block.attentions.0.norm")
    key = key.replace("time_embed_0", "time_embedding.linear_1")
    key = key.replace("time_embed_2", "time_embedding.linear_2")
    key = key.replace("label_emb_0_0", "add_embedding.linear_1")
    key = key.replace("label_emb_0_2", "add_embedding.linear_2")
    key = key.replace("out_0", "conv_norm_out")
    key = key.replace("out_2", "conv_out")
    return key

# modified from diffusers.loaders.lora_conversion_utils._maybe_map_sgm_blocks_to_diffusers
def dict_process_1(state_dict, unet_config, delimiter="_", block_slice_pos=5):

    # 1. get all state_dict_keys
    all_keys = list(state_dict.keys())
    sgm_patterns = ["input_blocks", "middle_block", "output_blocks"]

    # 2. check if needs remapping, if not return original dict
    is_in_sgm_format = False
    for key in all_keys:
        if any(p in key for p in sgm_patterns):
            is_in_sgm_format = True
            break

    if not is_in_sgm_format:
        return state_dict

    # 3. Else remap from SGM patterns
    new_state_dict = {}
    inner_block_map = ["resnets", "attentions", "upsamplers"]

    # Retrieves # of down, mid and up blocks
    input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()

    for layer in all_keys:
        if "text" in layer:
            new_state_dict[layer] = state_dict.pop(layer)
        else:
            if not any(p in layer for p in sgm_patterns):
                print("not process", layer)
                continue
            layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])

            if sgm_patterns[0] in layer:
                input_block_ids.add(layer_id)
            elif sgm_patterns[1] in layer:
                middle_block_ids.add(layer_id)
            elif sgm_patterns[2] in layer:
                output_block_ids.add(layer_id)
            else:
                raise ValueError(
                    f"Checkpoint not supported because layer {layer} not supported."
                )

    input_blocks = {
        layer_id: [
            key for key in state_dict if f"input_blocks{delimiter}{layer_id}" in key
        ]
        for layer_id in input_block_ids
    }
    middle_blocks = {
        layer_id: [
            key for key in state_dict if f"middle_block{delimiter}{layer_id}" in key
        ]
        for layer_id in middle_block_ids
    }
    output_blocks = {
        layer_id: [
            key for key in state_dict if f"output_blocks{delimiter}{layer_id}" in key
        ]
        for layer_id in output_block_ids
    }

    # Rename keys accordingly
    for i in input_block_ids:
        block_id = (i - 1) // (unet_config.layers_per_block + 1)
        if block_id == -1:
            block_id = 0
        layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)

        for key in input_blocks[i]:
            if block_slice_pos != len(key.split(delimiter)) - 1:
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
            else:
                # case: ['lora', 'unet', 'input', 'blocks', '0', '0.weight']
                inner_block_id = int(
                    key.split(delimiter)[block_slice_pos].split(".")[0]
                )

            inner_block_key = (
                inner_block_map[inner_block_id] if "op" not in key else "downsamplers"
            )
            inner_layers_in_block = str(layer_in_block_id) if "op" not in key else "0"
            if block_slice_pos != len(key.split(delimiter)) - 1:
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
            else:
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + ["."]
                    + [key.split(delimiter)[block_slice_pos].split(".")[-1]]
                )
            new_state_dict[new_key] = state_dict.pop(key)

    for i in middle_block_ids:
        key_part = None
        if i == 0:
            key_part = [inner_block_map[0], "0"]
        elif i == 1:
            key_part = [inner_block_map[1], "0"]
        elif i == 2:
            key_part = [inner_block_map[0], "1"]
        else:
            raise ValueError(f"Invalid middle block id {i}.")

        for key in middle_blocks[i]:
            new_key = delimiter.join(
                key.split(delimiter)[: block_slice_pos - 1]
                + key_part
                + key.split(delimiter)[block_slice_pos:]
            )
            new_state_dict[new_key] = state_dict.pop(key)

    for i in output_block_ids:
        block_id = i // (unet_config.layers_per_block + 1)
        layer_in_block_id = i % (unet_config.layers_per_block + 1)

        for key in output_blocks[i]:

            if block_slice_pos != len(key.split(delimiter)) - 1:
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
            else:
                # case: ['lora', 'unet', 'input', 'blocks', '0', '0.weight']
                inner_block_id = int(
                    key.split(delimiter)[block_slice_pos].split(".")[0]
                )

            inner_block_key = inner_block_map[inner_block_id]
            inner_layers_in_block = (
                str(layer_in_block_id) if inner_block_id < 2 else "0"
            )
            if block_slice_pos != len(key.split(delimiter)) - 1:
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
            else:
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + ["."]
                    + [key.split(delimiter)[block_slice_pos].split(".")[-1]]
                )

            new_state_dict[new_key] = state_dict.pop(key)

    if len(state_dict) > 0:
        raise ValueError("At this point all state dict entries have to be converted.")

    return new_state_dict

# modified from diffusers.loaders.lora_conversion_utils._convert_kohya_lora_to_diffusers
def dict_process_2(state_dict):

    unet_state_dict = {}

    lora_keys = [k for k in state_dict.keys()]
    for key in lora_keys:
        lora_name = key.split(".")[0]

        if lora_name.startswith("lora_unet_"):
            diffusers_name = key.replace("lora_unet_", "").replace("_", ".")

            if "input.blocks" in diffusers_name:
                diffusers_name = diffusers_name.replace("input.blocks", "down_blocks")
            else:
                diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")

            if "middle.block" in diffusers_name:
                diffusers_name = diffusers_name.replace("middle.block", "mid_block")
            else:
                diffusers_name = diffusers_name.replace("mid.block", "mid_block")
            if "output.blocks" in diffusers_name:
                diffusers_name = diffusers_name.replace("output.blocks", "up_blocks")
            else:
                diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")

            diffusers_name = diffusers_name.replace(
                "transformer.blocks", "transformer_blocks"
            )
            # diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
            # diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
            # diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
            # diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
            # diffusers_name = diffusers_name.replace("proj.in", "proj_in")
            # diffusers_name = diffusers_name.replace("proj.out", "proj_out")
            # diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")
            diffusers_name = diffusers_name.replace("to.q", "to_q")
            diffusers_name = diffusers_name.replace("to.k", "to_k")
            diffusers_name = diffusers_name.replace("to.v", "to_v")
            diffusers_name = diffusers_name.replace("to.out", "to_out")
            diffusers_name = diffusers_name.replace("proj.in", "proj_in")
            diffusers_name = diffusers_name.replace("proj.out", "proj_out")
            diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")
            # SDXL specificity.
            if "emb" in diffusers_name and "time.emb.proj" not in diffusers_name:
                pattern = r"\.\d+(?=\D*$)"
                diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
            if ".in." in diffusers_name:
                # diffusers_name = diffusers_name.replace("in.layers.2", "conv2")
                diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
                diffusers_name = diffusers_name.replace("in.layers.0", "norm1")
            if ".out." in diffusers_name:
                diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
                diffusers_name = diffusers_name.replace("out.layers.0", "norm2")
            if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
                diffusers_name = diffusers_name.replace("op", "conv")
            if "skip" in diffusers_name:
                diffusers_name = diffusers_name.replace(
                    "skip.connection", "conv_shortcut"
                )

            # # LyCORIS specificity.
            # if "time.emb.proj" in diffusers_name:
            #     diffusers_name = diffusers_name.replace(
            #         "time.emb.proj", "time_emb_proj"
            #     )
            # if "conv.shortcut" in diffusers_name:
            #     diffusers_name = diffusers_name.replace(
            #         "conv.shortcut", "conv_shortcut"
            #     )

            # General coverage.
            unet_state_dict[diffusers_name] = state_dict.pop(key)
            # if "transformer_blocks" in diffusers_name:
            #     unet_state_dict[diffusers_name] = state_dict.pop(key)
            #     # if "attn1" in diffusers_name or "attn2" in diffusers_name:
            #     # #     diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
            #     # #     diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
            #     #     unet_state_dict[diffusers_name] = state_dict.pop(key)
            #     # # elif "ff" in diffusers_name:
            #     # else:
            #     #     unet_state_dict[diffusers_name] = state_dict.pop(key)
            # elif any(key in diffusers_name for key in ("proj_in", "proj_out")):
            #     unet_state_dict[diffusers_name] = state_dict.pop(key)
            # else:
            #     unet_state_dict[diffusers_name] = state_dict.pop(key)

    if len(state_dict) > 0:
        raise ValueError(
            f"The following keys have not been correctly be renamed: \n\n {', '.join(state_dict.keys())}"
        )

    unet_state_dict = {
        # f"{unet_name}.{module_name}": params
        f"{module_name}": params
        for module_name, params in unet_state_dict.items()
    }

    new_state_dict = {**unet_state_dict}
    return new_state_dict
