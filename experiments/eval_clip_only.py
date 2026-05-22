"""
Lightweight eval - CLIP only, no SAM.
Fast to run, good for first results.
Run on Kaggle:
python eval_clip_only.py --data_root /kaggle/working/data/LoveDA --max_images 20
"""
import argparse, json, os, sys, time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geoprompt.prompt_templates import get_all_templates, LOVEDA_CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STRATEGIES = {
    "baseline":    lambda c: [f"a photo of {c}"],
    "template":    lambda c: [f"satellite view of {c}",
                               f"aerial photograph of {c}",
                               f"overhead imagery of {c}",
                               f"remote sensing image of {c}"],
    "ensemble":    lambda c: get_all_templates(c),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="/kaggle/working/data/LoveDA")
    p.add_argument("--max_images", type=int, default=20)
    p.add_argument("--clip_model", default="ViT-L/14")
    p.add_argument("--output_dir", default="/kaggle/working/results")
    return p.parse_args()

def collect_pairs(root, max_images=None):
    root = Path(root)
    pairs = []
    for img_dir in [root/"images_png", *root.rglob("images_png")]:
        lbl_dir = img_dir.parent / "masks_png"
        if img_dir.exists() and lbl_dir.exists():
            for img in sorted(img_dir.glob("*.png")):
                lbl = lbl_dir / img.name
                if lbl.exists():
                    pairs.append((str(img), str(lbl)))
    pairs = list(dict.fromkeys(pairs))
    return pairs[:max_images] if max_images else pairs

@torch.no_grad()
def segment_image(image, prompts_per_class, clip_model, preprocess,
                  patch_size=64, stride=32):
    img_np = np.array(image)
    h, w   = img_np.shape[:2]

    # Encode text
    all_texts, idx = [], {}
    for cls, prompts in prompts_per_class.items():
        idx[cls] = (len(all_texts), len(all_texts)+len(prompts))
        all_texts.extend(prompts)
    tokens     = clip.tokenize(all_texts).to(DEVICE)
    text_feats = F.normalize(clip_model.encode_text(tokens), dim=-1)
    cls_feats  = {c: text_feats[s:e].mean(0, keepdim=True)
                  for c, (s,e) in idx.items()}

    # Sliding window
    patch_list, positions = [], []
    for y0 in range(0, h-patch_size+1, stride):
        for x0 in range(0, w-patch_size+1, stride):
            patch_list.append(preprocess(
                Image.fromarray(img_np[y0:y0+patch_size, x0:x0+patch_size])))
            positions.append((y0, x0))

    BATCH = 128
    feats = []
    for i in range(0, len(patch_list), BATCH):
        t = torch.stack(patch_list[i:i+BATCH]).to(DEVICE)
        feats.append(F.normalize(clip_model.encode_image(t), dim=-1))
    feats = torch.cat(feats, 0)

    # Similarity maps
    sim_maps = {}
    for cls, tf in cls_feats.items():
        sims   = (feats @ tf.T).squeeze(-1).cpu().numpy()
        grid   = np.zeros((h, w), np.float32)
        count  = np.zeros((h, w), np.float32)
        for k, (y0, x0) in enumerate(positions):
            grid [y0:y0+patch_size, x0:x0+patch_size] += sims[k]
            count[y0:y0+patch_size, x0:x0+patch_size] += 1
        sim_maps[cls] = grid / np.maximum(count, 1)

    stack = np.stack([sim_maps[c] for c in LOVEDA_CLASSES], 0)
    exp   = np.exp(stack - stack.max(0, keepdims=True))
    return (exp / exp.sum(0, keepdims=True)).argmax(0)

def compute_miou(pred, gt, n=7, ignore=255):
    ious = []
    for c in range(n):
        p = pred==c; g = (gt==c)&(gt!=ignore)
        i = (p&g).sum(); u = (p|g).sum()
        ious.append(float(i)/float(u) if u>0 else float("nan"))
    valid = [x for x in ious if not np.isnan(x)]
    return ious, float(np.mean(valid)) if valid else 0.0

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = collect_pairs(args.data_root, args.max_images)
    print(f"Found {len(pairs)} pairs")
    if not pairs:
        return

    print(f"Loading CLIP {args.clip_model}...")
    model, preprocess = clip.load(args.clip_model, device=DEVICE)
    model.eval()

    all_results = {}
    for strat_name, template_fn in STRATEGIES.items():
        print(f"\n── {strat_name.upper()} ──────────────────────")
        ious_all, accs, times = [], [], []
        prompts = {c: template_fn(c) for c in LOVEDA_CLASSES}

        for img_path, lbl_path in tqdm(pairs):
            image = Image.open(img_path).convert("RGB")
            t0    = time.perf_counter()
            pred  = segment_image(image, prompts, model, preprocess)
            times.append(time.perf_counter() - t0)

            gt = np.array(Image.open(lbl_path)).astype(np.int64) - 1
            gt[gt < 0] = 255
            if pred.shape != gt.shape:
                pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST))

            ious, _ = compute_miou(pred, gt)
            ious_all.append(ious)
            valid = gt != 255
            accs.append(float((pred[valid]==gt[valid]).sum())/float(valid.sum()))

        per_class = np.nanmean(ious_all, axis=0)
        miou      = float(np.nanmean(per_class))
        result    = {
            "mIoU":           round(miou, 4),
            "pixel_accuracy": round(float(np.mean(accs)), 4),
            "mean_time_sec":  round(float(np.mean(times)), 3),
            "per_class_iou":  {c: round(float(v), 4)
                               for c, v in zip(LOVEDA_CLASSES, per_class)},
        }
        all_results[strat_name] = result
        print(f"  mIoU: {miou:.4f}  |  Pixel Acc: {result['pixel_accuracy']}")

    # Save + print table
    out = os.path.join(args.output_dir, "clip_only_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n══ RESULTS ═══════════════════════════════")
    print(f"{'Strategy':<14} {'mIoU':>7} {'PixAcc':>8} {'Time/s':>7}")
    print("─" * 36)
    for name, r in all_results.items():
        print(f"{name:<14} {r['mIoU']:>7.4f} "
              f"{r['pixel_accuracy']:>8.4f} {r['mean_time_sec']:>7.2f}")
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
