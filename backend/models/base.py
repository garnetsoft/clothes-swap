from abc import ABC, abstractmethod
from PIL import Image


class BaseSwapModel(ABC):
    @abstractmethod
    def load(self): ...

    @abstractmethod
    def unload(self): ...

    @abstractmethod
    def is_loaded(self) -> bool: ...


class GarmentSwapModel(BaseSwapModel):
    """Takes a person image + garment reference image, returns the swap."""
    @abstractmethod
    def swap(self, person: Image.Image, garment: Image.Image,
             mask: Image.Image, **kwargs) -> Image.Image: ...


class TextSwapModel(BaseSwapModel):
    """Takes a person image + text prompt, returns the swap."""
    @abstractmethod
    def swap(self, person: Image.Image, prompt: str,
             mask: Image.Image, **kwargs) -> Image.Image: ...
