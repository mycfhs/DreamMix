import copy
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
    rescale_noise_cfg,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from transformers import set_seed

import random

from utils import (
    add_fooocus_inpaint_patch,
    add_fooocus_inpaint_head_patch_with_work,
    sks_decompose,
    orthogonal_decomposition,
    KSampler,
)


import modules.anisotropic as anisotropic
import modules.inpaint_worker as inpaint_worker


def blur_guidance(latents, positive_x0, timestep, sharpness):
    # ! Fooocus trick
    # We implemented a carefully tuned variation of Section 5.1 of "Improving Sample Quality of Diffusion Models Using Self-Attention Guidance". The weight is set to very low, but this is Fooocus's final guarantee to make sure that the XL will never yield an overly smooth or plastic appearance (examples here). This can almost eliminate all cases for which XL still occasionally produces overly smooth results, even with negative ADM guidance. (Update 2023 Aug 18, the Gaussian kernel of SAG is changed to an anisotropic kernel for better structure preservation and fewer artifacts.)
    current_step = 1.0 - timestep.to(latents) / 999.0
    global_diffusion_progress = current_step.detach().cpu().numpy().tolist()

    positive_eps = latents - positive_x0
    alpha = 0.001 * sharpness * global_diffusion_progress

    positive_eps_degraded = anisotropic.adaptive_anisotropic_filter(
        x=positive_eps, g=positive_x0
    )
    positive_eps_degraded_weighted = positive_eps_degraded * alpha + positive_eps * (
        1.0 - alpha
    )

    return latents - positive_eps_degraded_weighted


def prepare_noise(latent_image, seed=None, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = None
    # if seed is not None:
    #     generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        )

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn(
            [1] + list(latent_image.size())[1:],
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        )
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class FooocusSDXLInpaintPipeline(StableDiffusionXLInpaintPipeline):
    def only_load_fooocus_unet_and_cover_pipe_unet_for_train(
        self, fooocus_model_path
    ):
        print(f"Loading fooocus unet from {fooocus_model_path} ...")
        # _device = self.device
        # self.unet = self.unet.to("cpu")
        add_fooocus_inpaint_patch(
            self.unet,
            model_path=fooocus_model_path,
        )

        print("Finish loading fooocus unet")

    def preload_fooocus_unet(
        self, fooocus_model_path, lora_configs=[], add_double_sa=False
    ):
        """
        lora_config: {
            path: scale, for_unet: bool, for_fooocus: bool
        }
        """
        if hasattr(self, "fooocus_unet"):
            print("fooocus_unet already loaded. Reloading.")
        print(f"Loading fooocus unet from {fooocus_model_path} ...")
        self.unload_lora_weights()
        _device = self.device
        self.unet = self.unet.to("cpu")

        self.fooocus_unet = copy.deepcopy(self.unet).to(_device)

        add_fooocus_inpaint_patch(
            self.fooocus_unet,
            model_path=fooocus_model_path,
        )
        print("fooocus unet loaded")

        if add_double_sa:
            self._add_double_sa(self.fooocus_unet)

        adapter_names_unet, adapter_names_fooocus = [], []
        adapter_scales_unet, adapter_scales_fooocus = [], []
        for lora_config in lora_configs:
            # scale, for_unet, for_fooocus = lora_setting
            # {"model_path": "./lora-dreambooth-model/pytorch_lora_weights.safetensors", "scale": 1, "for_unet": True, "for_fooocus_unet":True},
            assert (
                lora_config["for_fooocus_unet"] or lora_config["for_unet"]
            ), "lora_config should be for_fooocus_unet or for_unet or both"
            print(f"Loading lora... config: {lora_config} ...")
            adapter_name = lora_config["model_path"].replace(".", "_")

            if lora_config["for_raw_unet"]:
                self.load_lora_weights(
                    lora_config["model_path"], adapter_name=adapter_name
                )
                adapter_names_unet.append(adapter_name)
                adapter_scales_unet.append(lora_config["scale"])
            if lora_config["for_fooocus_unet"]:
                self.unet, self.fooocus_unet = self.fooocus_unet, self.unet
                self.load_lora_weights(
                    lora_config["model_path"], adapter_name=adapter_name
                )
                adapter_names_fooocus.append(adapter_name)
                adapter_scales_fooocus.append(lora_config["scale"])
                self.unet, self.fooocus_unet = self.fooocus_unet, self.unet

        self.unet, self.fooocus_unet = self.fooocus_unet, self.unet
        self.set_adapters(adapter_names_fooocus, adapter_weights=adapter_scales_fooocus)
        self.unet, self.fooocus_unet = self.fooocus_unet, self.unet

        print("lora loaded")
        self.fooocus_unet.to("cpu")
        self.unet = self.unet.to(_device)
        self.set_adapters(adapter_names_unet, adapter_weights=adapter_scales_unet)

        print("Finish loading fooocus unet")

    @torch.no_grad()
    def __call__(
        self,
        debug=False,
        decompose_prefix_prompt="",
        isf_global_time=-1,
        isf_global_ia = 1,
        soft_blending=False,
        sks_decompose_words=[],
        fooocus_model_head_path=None,
        fooocus_model_head_upscale_path=None,
        sharpness=2,
        fooocus_time=0.7,
        inpaint_respective_field=0.618,
        adm_scaler_positive=1,
        adm_scaler_negative=1,
        adm_scaler_end=0.0,
        seed=None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image=None,
        mask_image=None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 0.9999,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image=None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        assert hasattr(
            self, "fooocus_unet"
        ), "fooocus_unet not loaded. Use pipe.preload_fooocus_unet() first."


        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # ! load Fooocus model
        if seed is not None:
            SEED_LIMIT_NUMPY = 2**32
            seed = int(seed) % SEED_LIMIT_NUMPY
            set_seed(seed)
            seed_everything(seed)

        device = self.vae.device
        self.fooocu_unet = self.fooocus_unet.to("cpu")
        self.unet = self.unet.to(device)

        target_size = (height, width)
        image = image.resize(target_size)
        mask_image = mask_image.resize(target_size)

        image_for_inpaint_work = image.copy()
        mask_image_for_inpaint_work = mask_image.copy()

        inpaint_work = inpaint_worker.InpaintWorker(
            image=np.asarray(image),
            mask=np.asarray(mask_image)[:, :, 0],
            use_fill=strength > 0.99,
            k=inpaint_respective_field,
            path_upscale_models=fooocus_model_head_upscale_path,
        )

        if debug:
            raise NotImplementedError("debug mode not implemented yet")

        add_fooocus_inpaint_head_patch_with_work(
            self.fooocus_unet, self, fooocus_model_head_path, inpaint_work
        )
        self.fooocus_unet = self.fooocus_unet.to(device)


        # image = Image.fromarray(inpaint_work.interested_fill)
        image = Image.fromarray(inpaint_work.interested_image)
        mask_image = Image.fromarray(inpaint_work.interested_mask)
        # ! load Fooocus model end

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            mask_image,
            height,
            width,
            strength,
            callback_steps,
            output_type,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
            padding_mask_crop,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._denoising_start = denoising_start
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )



        # ! HHH sks_decompose
        prompt_embeds_decomposed = None
        if len(sks_decompose_words) > 0:
            decompose_words_num = len(sks_decompose_words)
            decompose_str = " ".join(sks_decompose_words)

            decompose_str = decompose_prefix_prompt + " " + decompose_str
            (
                sks_raw_prompt_embeds,
                _,
                pooled_sks_raw_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=decompose_str,
                prompt_2=decompose_str,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
            alpha = 0.0
            
            prompt_embeds_decomposed = prompt_embeds.clone()
            prompt_embeds_decomposed[0] = alpha * prompt_embeds[0] + (
                1 - alpha
            ) * sks_decompose(
                prompt,
                prompt_embeds[0],
                sks_raw_prompt_embeds[0],
                decompose_words_num,
                decompose_prefix_prompt,
            )
            prompt_embeds_decomposed_pooled = orthogonal_decomposition(
                pooled_prompt_embeds[0], pooled_sks_raw_prompt_embeds[0]
            ).unsqueeze(0)

        # 4. set timesteps
        def denoising_value_valid(dnv):
            return isinstance(dnv, float) and 0 < dnv < 1

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=(
                self.denoising_start
                if denoising_value_valid(self.denoising_start)
                else None
            ),
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )

        # 5. Preprocess mask and image
        image_latents = inpaint_work.latent
        mask_latent = inpaint_work.latent_mask

        ksampler = KSampler(image_latents, num_inference_steps, device)

        noise = prepare_noise(image_latents, seed=seed).to(device=device)
        if strength > 0.9999:
            noise = noise * torch.sqrt(1.0 + ksampler.sigmas[0] ** 2.0)
        else:
            noise = noise * ksampler.sigmas[0]

        latents = image_latents + noise
        # latents = noise

        # 8. Check that sizes of mask, masked image and latents match

        # 8.1 Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 10. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if prompt_embeds_decomposed is not None:
                prompt_embeds_decomposed = torch.cat([negative_prompt_embeds, prompt_embeds_decomposed], dim=0)
                add_text_embeds_pooled = torch.cat(
                    [negative_pooled_prompt_embeds, prompt_embeds_decomposed_pooled], dim=0
                )
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_neg_time_ids = add_neg_time_ids.repeat(
                batch_size * num_images_per_prompt, 1
            )
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)
        if prompt_embeds_decomposed is not None:
            prompt_embeds_decomposed = prompt_embeds_decomposed.to(device)
            prompt_embeds, prompt_embeds_decomposed = prompt_embeds_decomposed, prompt_embeds 

            add_text_embeds_pooled = add_text_embeds_pooled.to(device)
            add_text_embeds, add_text_embeds_pooled = add_text_embeds_pooled, add_text_embeds

        # ! Negative ADM guidance
        original_size_scaler = (
            original_size[0] * adm_scaler_positive,
            original_size[1] * adm_scaler_positive,
        )
        negative_original_size_scaler = (
            negative_original_size[0] * adm_scaler_negative,
            negative_original_size[1] * adm_scaler_negative,
        )
        add_time_ids_scaler, add_neg_time_ids_scaler = self._get_add_time_ids(
            original_size_scaler,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size_scaler,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids_scaler = add_time_ids_scaler.repeat(
            batch_size * num_images_per_prompt, 1
        )

        if self.do_classifier_free_guidance:
            add_neg_time_ids_scaler = add_neg_time_ids_scaler.repeat(
                batch_size * num_images_per_prompt, 1
            )
            add_time_ids_scaler = torch.cat(
                [add_neg_time_ids_scaler, add_time_ids_scaler], dim=0
            )
            add_time_ids_scaler = add_time_ids_scaler.to(device)
        # ! Negative ADM guidance end

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        if (
            self.denoising_end is not None
            and self.denoising_start is not None
            and denoising_value_valid(self.denoising_end)
            and denoising_value_valid(self.denoising_start)
            and self.denoising_start >= self.denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {self.denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {self.denoising_end} when using type float."
            )
        elif self.denoising_end is not None and denoising_value_valid(
            self.denoising_end
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        # 11.1 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        energy_generator = None

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for i, t in enumerate(timesteps):
            for i in range(num_inference_steps):
                if self.interrupt:
                    continue

                if i == isf_global_time:
                    def image_blending_toglobal(latents, inpaint_work, isf_global_ia=1):
                        latents = pred_x0
                        needs_upcasting = (self.vae.dtype == torch.float16 and self.vae.config.force_upcast)
                        if needs_upcasting:
                            self.upcast_vae()
                            latents = latents.to(
                                next(iter(self.vae.post_quant_conv.parameters())).dtype
                            )
                        latents = latents / self.vae.config.scaling_factor
                        image = self.vae.decode(latents, return_dict=False)[0]
                        image = self.image_processor.postprocess(image, output_type=output_type)
                        # image[0].save("test_inpaint_toglobal_before.png")
                        image = [np.array(x) for x in image]
                        image = [inpaint_work.post_process(x, soft_blending) for x in image]
                        image = [Image.fromarray(x) for x in image]
                        image = image[0]
                        # image.save("test_inpaint_toglobal_after.png")

                        if isf_global_ia < 1:
                            image = inpaint_worker.InpaintWorker(
                                image=np.asarray(image),
                                mask=np.asarray(mask_image_for_inpaint_work)[:, :, 0],
                                use_fill=False,
                                k=isf_global_ia,
                                path_upscale_models=fooocus_model_head_upscale_path,
                            ).interested_image
                            image = Image.fromarray(image)
                            # image.save("test_inpaint_toglobal_after_crop.png")
                        image = self.image_processor.preprocess(image).to(latents)
                        latents = self._encode_vae_image(image=image, generator=None)

                        # cast back to fp16 if needed
                        if needs_upcasting:
                            self.vae.to(dtype=torch.float16)
                        return latents

                    latents = image_blending_toglobal(latents, inpaint_work, isf_global_ia)
                    inpaint_work = inpaint_worker.InpaintWorker(
                        image=np.asarray(image_for_inpaint_work),
                        mask=np.asarray(mask_image_for_inpaint_work)[:, :, 0],
                        use_fill=False,
                        k=isf_global_ia,
                        path_upscale_models=fooocus_model_head_upscale_path,
                    )

                    ksampler = KSampler(latents, num_inference_steps, device)

                    sigma = ksampler.sigmas[i]
                    energy_sigma = sigma.reshape([1] + [1] * (len(latents.shape) - 1))
                    current_energy = torch.randn(
                        latents.size(), dtype=latents.dtype, generator=energy_generator, device="cpu").to(latents) * energy_sigma

                    latents = latents + current_energy

                    add_fooocus_inpaint_head_patch_with_work(
                        self.fooocus_unet,
                        self,
                        fooocus_model_head_path,
                        inpaint_work,
                    )
                    image_latents = inpaint_work.latent
                    mask_latent = inpaint_work.latent_mask
                    if prompt_embeds_decomposed is not None:
                        prompt_embeds, prompt_embeds_decomposed = prompt_embeds_decomposed, prompt_embeds
                        add_text_embeds, add_text_embeds_pooled = add_text_embeds_pooled, add_text_embeds

                t = ksampler.timestep(i)

                # ! fooocus add noise
                sigma = ksampler.sigmas[i]
                energy_sigma = sigma.reshape([1] + [1] * (len(latents.shape) - 1))
                current_energy = torch.randn(
                    latents.size(), dtype=latents.dtype, generator=energy_generator, device="cpu").to(latents) * energy_sigma

                latents = latents * mask_latent + (image_latents + current_energy) * (1.0 - mask_latent)

                # ! fooocus add noise end

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )

                latent_model_input = ksampler.calculate_input(i, latent_model_input).to(
                    dtype=self.fooocus_unet.dtype
                )

                #! Fooocus part
                if i <= int(num_inference_steps * adm_scaler_end):
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids_scaler,
                    }
                else:
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }

                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                # predict the noise residual
                # if i <= int(len(timesteps)*fooocus_time*strength):
                if i <= int(num_inference_steps * fooocus_time * strength):
                    # if fooocus_unet.device == torch.device("cpu"):  # save cuda memory
                    self.unet = self.unet.to("cpu")
                    self.fooocus_unet = self.fooocus_unet.to(device)

                    noise_pred = self.fooocus_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    # if self.unet.device == torch.device("cpu"):  # save cuda memory
                    self.fooocus_unet = self.fooocus_unet.to("cpu")
                    self.unet = self.unet.to(device)

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    positive_x0 = ksampler.calculate_denoised(
                        i, noise_pred_text, latents
                    )
                    negative_x0 = ksampler.calculate_denoised(
                        i, noise_pred_uncond, latents
                    )
                    if sharpness > 0:
                        positive_x0 = blur_guidance(latents, positive_x0, t, sharpness)

                    negative_eps = latents - negative_x0
                    positive_eps = latents - positive_x0

                    final_eps = negative_eps + self.guidance_scale * (
                        positive_eps - negative_eps
                    )
                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        final_eps = rescale_noise_cfg(
                            final_eps,
                            positive_eps,
                            guidance_rescale=self.guidance_rescale,
                        )
                    pred_x0 = latents - final_eps
                else:
                    pred_x0 = ksampler.calculate_denoised(i, noise_pred, latents)
                    if sharpness > 0:
                        pred_x0 = blur_guidance(latents, pred_x0, t, sharpness)

                # compute the previous noisy sample x_t -> x_t-1
                latents = ksampler.step(i, pred_x0, latents)
                #! Fooocus part end

                if (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            return StableDiffusionXLPipelineOutput(images=latents)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)


        image = [np.array(x) for x in image]
        image = [inpaint_work.post_process(x) for x in image]
        image = [Image.fromarray(x) for x in image]


        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
