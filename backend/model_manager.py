import gc
import torch
from .models.segmentation import SegmentationModel
from .models.vton import VTONModel
from .models.controlnet_sd import ControlNetSDModel


class ModelManager:
    """
    Owns all model instances and enforces the memory budget:
    - SegFormer stays resident (small, ~0.4 GB)
    - Only one swap model (VTON or ControlNet) loaded at a time
    - Unloads the previous swap model before loading the next
    """

    def __init__(self):
        self.seg = SegmentationModel()
        self._vton = VTONModel()
        self._controlnet = ControlNetSDModel()
        self._active_mode: str | None = None

    def ensure_ready(self, mode: str):
        if not self.seg.is_loaded():
            self.seg.load()

        if mode == "garment":
            if self._active_mode == "text" and self._controlnet.is_loaded():
                self._controlnet.unload()
                gc.collect()
                torch.cuda.empty_cache()
            if not self._vton.is_loaded():
                self._vton.load()

        elif mode == "text":
            if self._active_mode == "garment" and self._vton.is_loaded():
                self._vton.unload()
                gc.collect()
                torch.cuda.empty_cache()
            if not self._controlnet.is_loaded():
                self._controlnet.load()

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        self._active_mode = mode

    @property
    def vton(self) -> VTONModel:
        return self._vton

    @property
    def controlnet(self) -> ControlNetSDModel:
        return self._controlnet

    def status(self) -> dict:
        mem = {}
        if torch.cuda.is_available():
            mem = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb":  round(torch.cuda.memory_reserved() / 1e9, 2),
            }
        return {
            "active_mode":       self._active_mode,
            "seg_loaded":        self.seg.is_loaded(),
            "vton_loaded":       self._vton.is_loaded(),
            "controlnet_loaded": self._controlnet.is_loaded(),
            "memory":            mem,
        }


manager = ModelManager()
