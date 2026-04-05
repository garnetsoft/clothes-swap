from PIL import Image
from .base import TextSwapModel


class ControlNetSDModel(TextSwapModel):
    """
    Phase 2: SD 1.5 + ControlNet-OpenPose for text-prompt clothing swap.
    OpenPose conditioning preserves body pose while inpainting the clothing region.
    Stub — not yet implemented.
    """

    def load(self):
        raise NotImplementedError("Text-prompt mode is Phase 2 — not yet implemented.")

    def unload(self):
        pass

    def is_loaded(self) -> bool:
        return False

    def swap(self, person: Image.Image, prompt: str,
             mask: Image.Image, **kwargs) -> Image.Image:
        raise NotImplementedError("Text-prompt mode is Phase 2 — not yet implemented.")
