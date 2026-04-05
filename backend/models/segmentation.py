import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from ..utils.config import cfg

GARMENT_LABELS = {
    "upper": [4, 7],     # upper-clothes, coat
    "lower": [6, 12],    # pants, skirt
    "dress": [7],
    "all":   [4, 6, 7, 12],
}


class SegmentationModel:
    def __init__(self):
        self._processor = None
        self._model = None

    def load(self):
        repo = cfg.models.segmentation.repo
        self._processor = SegformerImageProcessor.from_pretrained(repo)
        self._model = SegformerForSemanticSegmentation.from_pretrained(repo)
        self._model.eval()

    def is_loaded(self) -> bool:
        return self._model is not None

    def segment(self, image: Image.Image, garment_type: str = "upper") -> Image.Image:
        """Returns a binary RGB mask (white = clothing region)."""
        label_ids = GARMENT_LABELS.get(garment_type, GARMENT_LABELS["upper"])

        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        upsampled = torch.nn.functional.interpolate(
            outputs.logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1).squeeze().numpy()

        mask = np.zeros_like(pred, dtype=np.uint8)
        for lid in label_ids:
            mask[pred == lid] = 255

        return Image.fromarray(mask).convert("RGB")
