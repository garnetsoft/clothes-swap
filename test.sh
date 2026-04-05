/mnt/ssd/envs/llm-orin-310/bin/python -c "
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import numpy as np

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting', torch_dtype=torch.float16
    ).to('cuda')
pipe.load_ip_adapter('h94/IP-Adapter', subfolder='models', weight_name='ip-adapter_sd15.bin')
pipe.enable_attention_slicing(1)
pipe.set_progress_bar_config(disable=True)
print('loaded ok')

dummy = Image.fromarray(np.zeros((512,512,3), dtype=np.uint8))
mask  = Image.fromarray(np.ones((512,512), dtype=np.uint8)*255, mode='L')
try:
    out = pipe(prompt='test', image=dummy, mask_image=mask,
               ip_adapter_image=[dummy], num_inference_steps=1).images[0]
    print('inference ok')
except Exception as e:
    print(f'error: {e}')
"
