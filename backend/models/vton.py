import gc
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from .base import GarmentSwapModel
from ..utils.config import cfg

# Upgrade path: replace this class body with CatVTON or OOTDiffusion
# while keeping the same load/unload/swap interface.
#
# CatVTON (zhengchong/CatVTON) — better VTON quality, needs their custom
# pipeline code. Swap in when ready:
#   from catvton_pipeline import CatVTONPipeline
#   self._pipe = CatVTONPipeline(base_ckpt=..., attn_ckpt=..., device=device)


class VTONModel(GarmentSwapModel):
    """
    Phase 1: IP-Adapter + SD 1.5 inpainting.
    The garment image conditions the inpainting via IP-Adapter image embedding.
    Preserves background and body shape outside the segmented clothing mask.
    """

    def __init__(self):
        self._pipe = None

    def load(self):
        m = cfg.models.vton
        device = cfg.device
        dtype = torch.float16 if cfg.dtype == "float16" else torch.float32

        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            m.base_repo, torch_dtype=dtype
        )
        self._pipe.load_ip_adapter(
            m.ip_adapter_repo,
            subfolder="models",
            weight_name=m.ip_adapter_weight,
        )
        self._pipe.set_ip_adapter_scale(m.ip_adapter_scale)
        self._pipe.enable_model_cpu_offload()
        self._pipe.enable_vae_tiling()
        self._pipe.set_progress_bar_config(disable=True)

    def unload(self):
        del self._pipe
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def swap(self, person: Image.Image, garment: Image.Image,
             mask: Image.Image, **kwargs) -> Image.Image:
        size = cfg.inference.image_size
        steps = kwargs.get("num_inference_steps", cfg.inference.num_inference_steps)
        guidance = kwargs.get("guidance_scale", cfg.inference.guidance_scale)
        seed = kwargs.get("seed", cfg.inference.seed)

        orig_size = person.size
        person_r  = person.resize((size, size), Image.LANCZOS)
        garment_r = garment.resize((size, size), Image.LANCZOS)
        mask_r    = mask.resize((size, size), Image.NEAREST)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            result = self._pipe(
                prompt="a person wearing clothes, high quality, photorealistic",
                negative_prompt="deformed, blurry, bad anatomy, disfigured, watermark",
                image=person_r,
                mask_image=mask_r.convert("L"),
                ip_adapter_image=[garment_r],
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            ).images[0]

        return result.resize(orig_size, Image.LANCZOS)
