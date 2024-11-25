# DreamMix: Decoupling Object Attributes for Enhanced Editability in Customized Image Inpainting

<!-- [![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/todo) -->

This repository contains the official implementation of the following paper:
> **DreamMix: Decoupling Object Attributes for Enhanced Editability in Customized Image Inpainting** <br>

<div align="center">
    <img src="./assets/teaser.png" alt="Teaser Image">
</div>

## Installation

Please note that our project requires 24GB of GPU memory to run.

```bash
git clone https://github.com/mycfhs/DreamMix.git
cd DreamMix
```

### 1. Prepare Environment

To set up our environment, please follow these instructions:

```bash
conda create -n DreamMix python=3.10
conda activate DreamMix
pip install -r requirements.txt
```

### 2. Download Checkpoints

Download the [fooocus model head patch](https://huggingface.co/lllyasviel/fooocus_inpaint/blob/main/inpaint_v26.fooocus.patch) and place it in `models/fooocus_inpaint`. Also, download the [upscale model](https://huggingface.co/metercai/SimpleSDXL/tree/main/upscale_models) and place it in `models/upscale_mode`.

### 3. Data Preparation

Install lang-sam:

```bash
# install lang-sam:
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
# ! modify: pyproject.toml  :  pillow 9.3.0->9.4.0
pip install -e .
```

Generate regular image using `make_img.ipynb`

Download DreamBooth dataset [here](https://github.com/google/dreambooth) and rename dir to "train_data".

### 4. Train with DreamMix

After downloading the base model, to execute user inference, use the following command:

```bash
CATEGORY="teapot"
accelerate launch train.py \
    --category="${CATEGORY}" \
    --output_dir="lora/${CATEGORY}" \
    --regular_dir="./regular_${CATEGORY}" \
    --regular_prob=0.3 \
    --loss_reweight_object=1.5 \
    --loss_reweight_background=0.6 \
    --pretrained_model_name_or_path="frankjoshua/juggernautXL_v8Rundiffusion"  \
    --instance_data_dir="train_data/${CATEGORY}/image" \
    --mixed_precision="no" \
    --instance_prompt="${categroy_prompt[${CATEGORY}]}" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --seed="0" \
    --checkpointing_steps=250 \
    --resume_from_checkpoint="latest" \
    --enable_xformers_memory_efficient_attention
```

### 5. Infer with DreamMix

Follow the instructions in infer.ipynb.
