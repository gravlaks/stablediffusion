import typing as T

import numpy as np
import torch
from torchvision.transforms.functional import resize, InterpolationMode
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import   PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler

from .custom_base_pipeline import CustomBasePipeline


class InpaintPipeline(CustomBasePipeline):
    def __init__(self, vae: AutoencoderKL, 
                       text_encoder: CLIPTextModel, 
                       tokenizer: CLIPTokenizer, 
                       unet: UNet2DConditionModel, 
                       scheduler: T.Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],) -> None:
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)


    @torch.no_grad()
    def __call__(self, prompt: T.Union[str, T.List[str]],
                       init_image: T.Union[torch.FloatTensor, np.ndarray],
                       # If input is tensor it expected to have [1, 3, h, w] in [-1, 1]
                       # if input is numpy it expected to have [h, w, 3] in [0, 255]
                       mask_image: T.Union[torch.FloatTensor, np.ndarray],
                       strength: float = 0.8,
                       num_inference_steps: int = 50,
                       guidance_scale: float = 7.5,
                       negative_prompt: T.Optional[T.Union[str, T.List[str]]] = None,
                       num_images_per_prompt: T.Optional[int] = 1,
                       eta: float = 0.0,
                       generator: T.Optional[torch.Generator] = None,
                       latents: T.Optional[torch.FloatTensor] = None,
                       output_type: str = 'numpy'):
        if output_type not in ['numpy', 'pil', 'full_steps']:
            raise ValueError(f'output type: {output_type} is not supported')
        if isinstance(init_image, np.ndarray):
            init_image = init_image[None, ...].transpose(0, 3, 1, 2)
            init_image = init_image.astype(np.float32) / 255.0
            init_image = torch.from_numpy(init_image) * 2.0 - 1.0
        if isinstance(mask_image, np.ndarray):
            mask_image = torch.from_numpy(mask_image)
        
        do_classifier_free_guidance = guidance_scale > 1.0
        # convert text to embeding
        text_embeddings, text_input = self.text_to_embeding(prompt, num_images_per_prompt)
        text_input_ids = text_input.input_ids
        max_length = text_input_ids.shape[-1]

        # configure negative prompt
        if do_classifier_free_guidance:
            negative_prompt = self.check_negative_prompt(prompt, negative_prompt)
            uncond_embeddings, _ = self.text_to_embeding(negative_prompt, num_images_per_prompt, max_length)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        #init_latent
        init_latents = self.initialize_latent_input(init_image,
                                                    text_embeddings.dtype, 
                                                    num_images_per_prompt=num_images_per_prompt, 
                                                    generator=generator)
        original_init_latents = init_latents

        mask = self.initialize_mask(mask_image,
                                    text_embeddings.dtype, 
                                    num_images_per_prompt=num_images_per_prompt)
        print(mask.shape, init_latents.shape)
        if not mask.shape == init_latents.shape:
            raise ValueError("The mask and init_image should be the same size!")

        # get the original timestep using init_timestep
        self.scheduler.set_timesteps(num_inference_steps)
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * self.batch_size * num_images_per_prompt, device=self.device)

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device, dtype=text_embeddings.dtype)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        latents = init_latents

        t_start = max(num_inference_steps - init_timestep + offset, 0)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps[t_start:].to(self.device)
        
        latents_steps = []
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            latents = self.diffusion_step(latents=latents,
                                          text_embeddings=text_embeddings,
                                          t = t,
                                          do_classifier_free_guidance=do_classifier_free_guidance,
                                          guidance_scale = guidance_scale, 
                                          eta = eta)

            init_latents_proper = self.scheduler.add_noise(original_init_latents, noise, torch.tensor([t]))
            latents = (init_latents_proper * mask) + (latents * (1 - mask))
            latents_steps.append(latents)
        
        # outputs:
        last_image_tensor = self.vae_decode(latents_steps[-1])
        if output_type == 'pil':
            return self.RGB_tensor_to_numpy(last_image_tensor, True)
        elif output_type == 'numpy':
            return self.RGB_tensor_to_numpy(last_image_tensor, False)
        elif output_type == 'full_steps':
            image_list = []
            for i, _ in enumerate(self.progress_bar(timesteps_tensor)):
                image_tensor = last_image_tensor = self.vae_decode(latents_steps[i])
                image_list.append(self.RGB_tensor_to_numpy(image_tensor, False))
            return np.concatenate(image_list, axis=0)


    @torch.no_grad()
    def initialize_latent_input(self, init_image: torch.Tensor,
                                      latents_dtype: T.Any,
                                      num_images_per_prompt: int = 1, 
                                      generator: T.Optional[torch.Generator] = None,) -> torch.Tensor:
        # encode the init image into latents and scale the latents
        init_image = init_image.to(device=self.device, dtype=latents_dtype)
        init_latent_dist = self.vae.encode(init_image).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        # expand init_latents for batch_size
        init_latents = torch.cat([init_latents] * self.batch_size * num_images_per_prompt, dim=0)

        return init_latents

    
    @torch.no_grad()
    def initialize_mask (self, mask_image: torch.Tensor,
                               latents_dtype: T.Any,
                               num_images_per_prompt: int = 1, ):
        w, h = mask_image.shape
        mask_image = resize(mask_image[None, None, ...], (h // 8, w // 8), InterpolationMode.NEAREST)
        mask_image = mask_image.tile(1,4,1,1)
        mask_image = mask_image.to(device=self.device, dtype=latents_dtype)
        mask = torch.cat([mask_image] * self.batch_size * num_images_per_prompt, dim=0)
        return 1 - mask