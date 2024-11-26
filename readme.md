# **DreamMix: Decoupling Object Attributes for Enhanced Editability in Customized Image Inpainting**


<!-- [![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/todo) -->

<div align="center">
    <img src="./assets/teaser.png" alt="Teaser Image" style="max-width: 100%; border-radius: 10px;">
</div>

---

## 📰 **News and Updates**  
- **[24.11.24]** Official release of code and paper!

This repository hosts the **official implementation** of our paper:

> **DreamMix: Decoupling Object Attributes for Enhanced Editability in Customized Image Inpainting**

---

## **Features**
- **Seamless Image Inpainting**: Decouple object attributes for enhanced control and editability.
- **Customizable Training**: Train with DreamBooth and fine-tune models for your needs.
- **User-Friendly Tools**: Straightforward inference pipeline with advanced inpainting tricks.

---

## 🚀 **Installation**

> **Note:** DreamMix requires a GPU with **24GB memory** to run.

1. Clone the repository:

    ```bash
    git clone https://github.com/mycfhs/DreamMix.git
    cd DreamMix
    ```

2. Prepare the environment:

    ```bash
    conda create -n DreamMix python=3.10
    conda activate DreamMix
    pip install -r requirements.txt
    ```

3. Download the necessary models:

    - [Fooocus inpaint v26 patch](https://huggingface.co/lllyasviel/fooocus_inpaint/blob/main/inpaint_v26.fooocus.patch) → Place in `models/fooocus_inpaint`.
    - [Upscale model](https://huggingface.co/metercai/SimpleSDXL/tree/main/upscale_models) → Place in `models/upscale_mode`.

---

### **Data Preparation**

1. Install **lang-segment-anything**:

    ```bash
    git clone https://github.com/mycfhs/lang-segment-anything
    cd lang-segment-anything
    pip install -e .
    ```

2. Generate regular images using the `make_img.ipynb` notebook.

3. Download the DreamBooth dataset [here](https://github.com/google/dreambooth) and rename the directory to `train_data`.

---

### **Training with DreamMix**

To begin training, use the following command:

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

---

### **Inference with DreamMix**

To perform inference, follow the instructions in `infer.ipynb`.

---

## 📜 **BibTeX**

If you find our work helpful, please consider giving us a ⭐ or citing our paper:

```bibtex
@article{shi2023dragdiffusion,
  title={DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing},
  author={Shi, Yujun and Xue, Chuhui and Pan, Jiachun and Zhang, Wenqing and Tan, Vincent YF and Bai, Song},
  journal={arXiv preprint arXiv:2306.14435},
  year={2023}
}
```

---

## 🙏 **Acknowledgements**

We extend our gratitude to the incredible open-source community. Our work is based on the following resources:

- [Fooocus](https://github.com/lllyasviel/Fooocus) Thanks their fantastic inpaint method [Inpaint v26 Fooocus Patch](https://huggingface.co/lllyasviel/fooocus_inpaint).
- We use [JuggernautXL v8 Rundiffusion](https://huggingface.co/frankjoshua/juggernautXL_v8Rundiffusion) as our base generator.

- Training code is based on [Diffusers SDXL DreamBooth example](https://github.com/huggingface/diffusers/blob/v0.30.2/examples/dreambooth/train_dreambooth_lora_sdxl.py).
- Image samples are collected from [Pixabay](https://pixabay.com/) and [COCO Dataset](https://cocodataset.org/).

Here's the techniques we have incorporated in Fooocus inpainting:
- **Blur Guidance**: Controlled with the `sharpness` parameter.
- **ADM Scaler**: Parameters `adm_scaler_positive, adm_scaler_negative, adm_scaler_end`
- **Inpaint Worker**: Enhanced inpainting logic.
- **Prompt Style Enhancement**: Improves prompt-adaptability.
- **Advanced Sampler & Scheduler**: `Dpmpp2mSdeGpuKarras`.
- **Hybrid Models**: Utilizes both base and inpainting models across different timesteps (`fooocus_time`).

---

## 📧 **Contact**

For questions or feedback, please reach out to us at **mycf2286247133@gmail.com**.

---

## 🔗 **Related Links**
- [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)
