from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .models import build_model, count_parameters
from .segmentation import segment_leaf


class Predictor:
    def __init__(
        self,
        checkpoint_path: str,
        class_map_path: str,
        image_size: int = 224,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        with open(class_map_path, "r", encoding="utf-8") as f:
            self.idx_to_class = {int(k): v for k, v in json.load(f).items()}

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name = ckpt.get("model", "efficientnet")
        self.num_classes = ckpt.get("num_classes", len(self.idx_to_class))
        self.image_size = ckpt.get("image_size", image_size)

        model = build_model(self.model_name, self.num_classes, use_pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        self.model = model

        self.checkpoint_path = checkpoint_path
        self.parameters = count_parameters(model)
        self.checkpoint_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)

        self.tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, use_segmentation: bool = True) -> Tuple[str, float, Image.Image]:
        proc_img = image.convert("RGB")
        if use_segmentation:
            proc_img, _ = segment_leaf(proc_img)
        x = self.tf(proc_img).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        return self.idx_to_class[int(idx.item())], float(conf.item()), proc_img

    @torch.no_grad()
    def predict_topk(
        self, image: Image.Image, k: int = 5, use_segmentation: bool = True
    ) -> Tuple[List[Tuple[str, float]], Image.Image, "torch.Tensor"]:
        proc_img = image.convert("RGB")
        if use_segmentation:
            proc_img, _ = segment_leaf(proc_img)
        x = self.tf(proc_img).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        k = min(k, probs.shape[0])
        confs, idxs = torch.topk(probs, k=k)
        topk = [(self.idx_to_class[int(i)], float(c)) for c, i in zip(confs.tolist(), idxs.tolist())]
        return topk, proc_img, probs.cpu()
