"""Evaluate GeoPrompt on LoveDA dataset."""
import argparse, json, os, time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

LOVEDA_CLASSES = ["background","building","road","water","barren","forest","agricultural"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="ensemble",
                   choices=["template","hierarchical","ensemble","context_aware"])
    p.add_argument("--clip_model",      default="ViT-L/14")
    p.add_argument("--sam_checkpoint",  default="checkpoints/sam_vit_h.pth")
    p.add_argument("--data_root",       default="data/LoveDA")
    p.add_argument("--split",           default="val")
    p.add_argument("--max_images",      type=int, default=None)
    p.add_argument("--output_dir",      default="results/loveda")
    return p.parse_args()

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

def compute_pixel_acc(pred, gt, ignore=255):
    valid = gt != ignore
    return float((pred[valid] == gt[valid]).sum()) / float(valid.sum())

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from geoprompt import GeoPromptSegmenter
    seg = GeoPromptSegmenter(args.clip_model, args.sam_checkpoint, args.strategy)

    img_dir = Path(args.data_root) / args.split / "images_png"
    lbl_dir = Path(args.data_root) / args.split / "masks_png"
    if not img_dir.exists():
        print(f"Data not found at {img_dir}")
        return

    paths = sorted(img_dir.glob("*.png"))
    if args.max_images:
        paths = paths[:args.max_images]

    ious_all, accs, times = [], [], []
    for img_path in tqdm(paths, desc=args.strategy):
        lbl_path = lbl_dir / img_path.name
        t0 = time.perf_counter()
        masks, _ = seg.segment(str(img_path), classes=LOVEDA_CLASSES, return_sim_maps=True)
        times.append(time.perf_counter() - t0)
        h, w = next(iter(masks.values())).shape
        pred = np.zeros((h, w), dtype=np.int64)
        for cls, mask in masks.items():
            pred[mask] = LOVEDA_CLASSES.index(cls)
        if lbl_path.exists():
            gt = np.array(Image.open(lbl_path)).astype(np.int64) - 1
            gt[gt < 0] = 255
            if pred.shape != gt.shape:
                from PIL import Image as PI
                pred = np.array(PI.fromarray(pred.astype(np.uint8)).resize(
                    (gt.shape[1], gt.shape[0]), PI.NEAREST))
            ious, _ = compute_miou(pred, gt, len(LOVEDA_CLASSES))
            ious_all.append(ious)
            accs.append(compute_pixel_acc(pred, gt))

    per_class = np.nanmean(ious_all, axis=0) if ious_all else []
    miou = float(np.nanmean(per_class)) if len(per_class) else 0.0
    results = {
        "strategy": args.strategy,
        "mIoU": round(miou, 4),
        "pixel_accuracy": round(float(np.mean(accs)) if accs else 0, 4),
        "mean_time_sec": round(float(np.mean(times)), 3),
        "per_class_iou": {c: round(float(v), 4)
                          for c, v in zip(LOVEDA_CLASSES, per_class)},
    }
    out = os.path.join(args.output_dir, f"results_{args.strategy}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nmIoU: {miou:.4f}  |  saved to {out}")
    return results

if __name__ == "__main__":
    main()
