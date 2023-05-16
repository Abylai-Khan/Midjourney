from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import torch
import PIL
from PIL import Image
from typing import Optional, Union, List, Callable, Dict, Any

from diffusers import StableUnCLIPImg2ImgPipeline, ImagePipelineOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import randn_tensor, PIL_INTERPOLATION


def center_resize_crop(image, size=224):
    w, h = image.size
    if h < w:
        h, w = size, size * w // h
    else:
        h, w = size * h // w, size

    image = image.resize((w, h))

    box = ((w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2)
    return image.crop(box)


def encode_image(image, pipe):
    device = pipe._execution_device
    dtype = next(pipe.image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = pipe.feature_extractor(
            images=image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    image_embeds = pipe.image_encoder(image).image_embeds

    return image_embeds


def generate_latents(pipe):
    shape = (1, pipe.unet.in_channels, pipe.unet.config.sample_size,
             pipe.unet.config.sample_size)
    device = pipe._execution_device
    dtype = next(pipe.image_encoder.parameters()).dtype

    return torch.randn(shape, device=device, dtype=dtype)
def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * \
          low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


class StableRemixImageProcessor(VaeImageProcessor):
    def __init__(self, w, h):
        super().__init__()
        self.w = w
        self.h = h

    def resize(self, image):
        image = center_resize_crop(image, self.w)
        return image

    def preprocess(self, image):
        image = super().preprocess(image)
        return image


class StableRemix(StableUnCLIPImg2ImgPipeline):
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None,
                        noise=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i: i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0",
                      deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        if noise is None:
            noise = randn_tensor(shape, generator=generator,
                                 device=device, dtype=dtype)

        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def _encode_image(
            self,
            image,
            device,
            batch_size,
            num_images_per_prompt,
            do_classifier_free_guidance,
            noise_level,
            generator,
            image_embeds,
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        if isinstance(image, PIL.Image.Image):
             repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        if image_embeds is None:
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeds = self.image_encoder(image).image_embeds

        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )

        image_embeds = image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, repeat_by, 1)
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        image_embeds = image_embeds.squeeze(1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        return image_embeds

    @torch.no_grad()
    def __call__(
            self,
            image: Union[torch.FloatTensor, PIL.Image.Image] = None,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 40,
            guidance_scale: float = 10,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[
                int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            noise_level: int = 0,
            image_embeds=None,
            timestemp=0,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is None and prompt_embeds is None:
            prompt = len(image) * [""] if isinstance(image, list) else ""

        self.check_inputs(
            prompt=prompt,
            image=None,
            height=height,
            width=width,
            callback_steps=callback_steps,
            noise_level=noise_level,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        noise_level = torch.tensor([noise_level], device=device)
        image_embeds = self._encode_image(
            image=None,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
            image_embeds=image_embeds,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latent_timestep = timesteps[timestemp:timestemp +
                                              1].repeat(batch_size * num_images_per_prompt)

        image_processor = StableRemixImageProcessor(width, height)
        image = image_processor.preprocess(image)

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            image=image,
            timestep=latent_timestep,
            batch_size=batch_size,
            dtype=prompt_embeds.dtype,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            generator=generator,
            noise=latents
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(self.progress_bar(timesteps[timestemp:])):
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                             (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        image = self.decode_latents(latents)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


def run_remixing(pipe, content_img, style_img, alphas, **kwargs):
    images = []

    content_emb = encode_image(content_img, pipe)
    style_emb = encode_image(style_img, pipe)

    for alpha in alphas:
        emb = slerp(alpha, content_emb, style_emb)
        image = pipe(image=content_img, image_embeds=emb, **kwargs).images[0]
        images.append(image)

    return images


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('content_img', type=Path, help='Path to content image')
    parser.add_argument('style_img', type=Path, help='Path to style image')
    parser.add_argument('--device', type=torch.device,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                        help='Which device to use ("cpu", "cuda", "cuda:1", ...)')
    parser.add_argument('save_dir', type=Path, nargs='?', default=Path('.'),
                        help='Path to dir where to save remixes')

    return parser.parse_args()


def main():
    args = parse_args()
    print('Using device:', args.device)

    pipe = StableRemix.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
    )
    pipe = pipe.to(args.device)
    pipe.enable_xformers_memory_efficient_attention()

    content_img = Image.open(args.content_img).convert('RGB')
    style_img = Image.open(args.style_img).convert('RGB')

    images = run_remixing(pipe, content_img, style_img, [0.6, 0.65, 0.7])
    for idx, image in enumerate(images):
        path = args.save_dir / f'remix_{idx}.png'
        print('Saving remix to', path)
        image.save(path)


if __name__ == '__main__':
    main()