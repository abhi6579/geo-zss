"""
Ablation study: compare all GeoPrompt strategies on LoveDA.
Run: python experiments/ablation_study.py --data_root /kaggle/input/loveda --max_images 50
"""
import argparse, json, os, sys, time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOVEDA_CLASSES = ["background","building","road","water","barren","forest","agricultural"]

STRATEGIES = [
    {"name": "baseline",      "strategy": "template",      "n_templates": 1},
    {"name": "template",      "strategy": "template",      "n_templates": 8},
    {"name": "hierarchical",  "strategy": "hierarchical",  "n_templates": -1},
    {"name": "ensemble",      "strategy": "ensemble",      "n_templates": -1},
    {"name": "context_aware", "strategy": "context_aware", "n_templates": -1},
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default="data/LoveDA")
    p.add_argument("--clip_model",     default="ViT-L/14")
    p.add_argument("--sam_checkpoint", default="checkpoints/sam_vit_h.pth")
    p.add_argument("--max_images",     type=int, default=50)
    p.add_argument("--output_dir",     default="results/ablation")
    return p.parse_args()

def collect_pairs(root, split="Val", max_images=None):
    pairs = []
    for scene in ["Urban", "Rural"]:
        img_dir = Path(root) / split / scene / "images_png"
        lbl_dir = Path(root) / split / scene / "masks_png"
        if not img_dir.exists():
            img_dir = Path(root) / split.lower() / scene.lower() / "images_png"
            lbl_dir = Path(root) / split.lower() / scene.lower() / "masks_png"
        if img_dir.exists():
            for img in sorted(img_dir.glob("*.png")):
                lbl = lbl_dir / img.name
                if lbl.exists():
                    pairs.append((str(img), str(lbl)))
    return pairs[:max_images] if max_images else pairs

def compute_miou(pred, gt, num_classes, ignore=255):
    ious = []
    for c in range(num_classes):
        p = pred == c
        g = (gt == c) & (gt != ignore)
        inter = (p & g).sum()
        union = (p | g).sum()
        ious.append(float(inter)/float(union) if union > 0 else float("nan"))
    valid = [x for x in ious if not np.isnan(x)]
    return ious, float(np.mean(valid)) if valid else 0.0

def run_strategy(cfg, pairs, clip_model, sam_checkpoint):
    from geoprompt import GeoPromptSegmenter
    seg = GeoPromptSegmenter(clip_model, sam_checkpoint, cfg["strategy"])

    ious_all, accs, times = [], [], []
    for img_path, lbl_path in tqdm(pairs, desc=cfg["name"]):
        t0 = time.perf_counter()
        masks, _ = seg.segment(img_path, classes=LOVEDA_CLASSES, return_sim_maps=True)
        times.append(time.perf_counter() - t0)

        h, w = next(iter(masks.values())).shape
        pred = np.zeros((h, w), dtype=np.int64)
        for cls, mask in masks.items():
            pred[mask] = LOVEDA_CLASSES.index(cls)

        gt = np.array(Image.open(lbl_path)).astype(np.int64) - 1
        gt[gt < 0] = 255
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST))

        ious, _ = compute_miou(pred, gt, len(LOVEDA_CLASSES))
        ious_all.append(ious)
        valid = gt != 255
        accs.append(float((pred[valid] == gt[valid]).sum()) / float(valid.sum()))

    per_class = np.nanmean(ious_all, axis=0)
    return {
        "name":           cfg["name"],
        "strategy":       cfg["strategy"],
        "mIoU":           round(float(np.nanmean(per_class)), 4),
        "pixel_accuracy": round(float(np.mean(accs)), 4),
        "mean_time_sec":  round(float(np.mean(times)), 3),
        "per_class_iou":  {c: round(float(v), 4)
                           for c, v in zip(LOVEDA_CLASSES, per_class)},
    }

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = collect_pairs(args.data_root, max_images=args.max_images)
    if not pairs:
        print(f"No data found at {args.data_root}")
        return

    print(f"Running {len(STRATEGIES)} strategies on {len(pairs)} images\n")
    all_results = []

    for cfg in STRATEGIES:
        print(f"\n── {cfg['name'].upper()} ──────────────────────")
        result = run_strategy(cfg, pairs, args.clip_model, args.sam_checkpoint)
        all_results.append(result)
        print(f"  mIoU: {result['mIoU']}  |  Pixel Acc: {result['pixel_accuracy']}")

    # Save
    out = os.path.join(args.output_dir, "ablation_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print table
    print("\n\n══ ABLATION SUMMARY ════════════════════════════")
    print(f"{'Strategy':<16} {'mIoU':>7} {'PixAcc':>8} {'Time/s':>7}")
    print("─" * 42)
    for r in all_results:
        print(f"{r['name']:<16} {r['mIoU']:>7.4f} {r['pixel_accuracy']:>8.4f} {r['mean_time_sec']:>7.2f}")
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
