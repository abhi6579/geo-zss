"""Scene context classifier for context-aware prompt selection."""
import numpy as np
from typing import Optional

SCENE_TYPES = ["urban", "rural", "coastal", "desert", "forest", "agricultural", "mixed"]

SCENE_DESCRIPTIONS = {
    "urban":        ["dense city from satellite", "urban buildings and roads overhead"],
    "rural":        ["rural countryside from satellite", "farmland and fields overhead"],
    "coastal":      ["coastal area from satellite", "shoreline and ocean overhead"],
    "desert":       ["desert landscape from satellite", "arid land overhead"],
    "forest":       ["dense forest from satellite", "tree canopy overhead"],
    "agricultural": ["agricultural land from satellite", "crop fields overhead"],
    "mixed":        ["mixed land use from satellite", "varied landscape overhead"],
}


class SceneContextClassifier:
    def __init__(self, device: Optional[str] = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._preprocess = None
        self._text_feats = None

    def _load(self):
        if self._model is None:
            import clip, torch, torch.nn.functional as F
            self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
            self._model.eval()
            all_texts, self._scene_idx = [], {}
            for scene, descs in SCENE_DESCRIPTIONS.items():
                self._scene_idx[scene] = (len(all_texts), len(all_texts) + len(descs))
                all_texts.extend(descs)
            tokens = clip.tokenize(all_texts).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = F.normalize(feats, dim=-1)
            scene_feats = []
            for scene in SCENE_TYPES:
                s, e = self._scene_idx[scene]
                scene_feats.append(feats[s:e].mean(dim=0))
            self._text_feats = torch.stack(scene_feats, dim=0)

    def classify(self, image) -> str:
        try:
            import torch, torch.nn.functional as F
            self._load()
            img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = self._model.encode_image(img_tensor)
                img_feat = F.normalize(img_feat, dim=-1)
            sims = (img_feat @ self._text_feats.T).squeeze(0).cpu().numpy()
            return SCENE_TYPES[int(sims.argmax())]
        except Exception as e:
            print(f"[SceneContextClassifier] Failed ({e}), using 'mixed'")
            return "mixed"


def heuristic_scene_type(image) -> str:
    img = np.array(image.convert("RGB")).astype(float) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    ndvi  = (g - r) / (g + r + 1e-6)
    water = (b - r) / (b + r + 1e-6)
    gray  = 1.0 - np.std(img, axis=-1).mean()
    if water.mean() > 0.1:   return "coastal"
    if ndvi.mean()  > 0.25:  return "forest"
    if ndvi.mean()  > 0.15:  return "agricultural"
    if gray         > 0.85:  return "urban"
    return "mixed"
