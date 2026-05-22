"""GeoPromptSegmenter: Main inference class."""
from __future__ import annotations
import os
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

from .prompt_templates import get_all_templates, get_hierarchical_templates, context_modified_templates if False else get_all_templates
from .prompt_templates import get_all_templates, get_hierarchical_templates, LOVEDA_CLASSES
from .ensemble import EnsembleAggregator
from .context_selector import SceneContextClassifier


class GeoPromptSegmenter:
    def __init__(self, clip_model: str = "ViT-L/14",
                 sam_checkpoint: str = "checkpoints/sam_vit_h.pth",
                 prompt_strategy: str = "ensemble",
                 device: Optional[str] = None):
        self.clip_model_name = clip_model
        self.sam_checkpoint  = sam_checkpoint
        self.prompt_strategy = prompt_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = None
        self._clip_preprocess = None
        self._sam_predictor = None
        self.aggregator = EnsembleAggregator()
        self.context_classifier = SceneContextClassifier()

    def _load_clip(self):
        if self._clip_model is None:
            import clip
            print(f"[GeoPrompt] Loading CLIP {self.clip_model_name}")
            self._clip_model, self._clip_preprocess = clip.load(
                self.clip_model_name, device=self.device)
            self._clip_model.eval()

    def _load_sam(self):
        if self._sam_predictor is None:
            from segment_anything import sam_model_registry, SamPredictor
            p = Path(self.sam_checkpoint).name.lower()
            model_type = "vit_h" if "vit_h" in p else "vit_l" if "vit_l" in p else "vit_b"
            sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
            sam.to(self.device)
            self._sam_predictor = SamPredictor(sam)

    def _build_prompts(self, classes: List[str],
                       image=None) -> Dict[str, List[str]]:
        strategy = self.prompt_strategy
        if strategy == "template":
            return {cls: get_all_templates(cls)[:8] for cls in classes}
        elif strategy == "hierarchical":
            return {cls: get_hierarchical_templates(cls) for cls in classes}
        elif strategy == "context_aware":
            scene = self.context_classifier.classify(image) if image else "mixed"
            return {cls: get_all_templates(cls) for cls in classes}
        else:  # ensemble
            return {cls: get_all_templates(cls) for cls in classes}

    @torch.no_grad()
    def _clip_similarity_map(self, image: Image.Image,
                              prompts_per_class: Dict[str, List[str]],
                              patch_size: int = 32) -> Dict[str, np.ndarray]:
        import clip as openai_clip
        self._load_clip()
        img_w, img_h = image.size
        all_texts, class_indices = [], {}
        for cls, prompts in prompts_per_class.items():
            class_indices[cls] = (len(all_texts), len(all_texts) + len(prompts))
            all_texts.extend(prompts)
        tokens = openai_clip.tokenize(all_texts).to(self.device)
        text_feats = self._clip_model.encode_text(tokens)
        text_feats = F.normalize(text_feats, dim=-1)
        class_text_feats = {}
        for cls, (s, e) in class_indices.items():
            class_text_feats[cls] = text_feats[s:e].mean(dim=0, keepdim=True)
        stride = patch_size // 2
        img_np = np.array(image)
        patch_feats, positions = [], []
        for y0 in range(0, img_h - patch_size + 1, stride):
            for x0 in range(0, img_w - patch_size + 1, stride):
                patch = Image.fromarray(img_np[y0:y0+patch_size, x0:x0+patch_size])
                t = self._clip_preprocess(patch).unsqueeze(0).to(self.device)
                feat = self._clip_model.encode_image(t)
                patch_feats.append(F.normalize(feat, dim=-1))
                positions.append((y0, x0))
        patch_feats = torch.cat(patch_feats, dim=0)
        sim_maps = {}
        for cls, t_feat in class_text_feats.items():
            sims = (patch_feats @ t_feat.T).squeeze(-1).cpu().numpy()
            grid  = np.zeros((img_h, img_w), dtype=np.float32)
            count = np.zeros_like(grid)
            for k, (y0, x0) in enumerate(positions):
                grid [y0:y0+patch_size, x0:x0+patch_size] += sims[k]
                count[y0:y0+patch_size, x0:x0+patch_size] += 1
            sim_maps[cls] = grid / np.maximum(count, 1)
        return sim_maps

    def segment(self, image_path: str, classes=None,
                return_sim_maps: bool = False):
        image   = Image.open(image_path).convert("RGB")
        classes = classes or LOVEDA_CLASSES
        prompts = self._build_prompts(classes, image)
        sim_maps = self._clip_similarity_map(image, prompts)
        all_maps = np.stack([sim_maps[c] for c in classes], axis=0)
        exp_maps = np.exp(all_maps - all_maps.max(axis=0, keepdims=True))
        prob_maps = exp_maps / exp_maps.sum(axis=0, keepdims=True)
        pred = prob_maps.argmax(axis=0)
        masks = {cls: pred == i for i, cls in enumerate(classes)}
        return (masks, sim_maps) if return_sim_maps else masks
