from PIL import Image
from .model_manager import manager


class SwapPipeline:
    def run_garment_swap(self, person: Image.Image, garment: Image.Image,
                         garment_type: str = "upper") -> Image.Image:
        manager.ensure_ready("garment")
        mask = manager.seg.segment(person, garment_type)
        return manager.vton.swap(person, garment, mask)

    def run_text_swap(self, person: Image.Image, prompt: str,
                      garment_type: str = "upper") -> Image.Image:
        manager.ensure_ready("text")
        mask = manager.seg.segment(person, garment_type)
        return manager.controlnet.swap(person, prompt, mask)

    def status(self) -> dict:
        return manager.status()


pipeline = SwapPipeline()
