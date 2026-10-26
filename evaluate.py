import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from src.data import find_mvtec_category, get_test_samples, image_transform
from src.model import PatchAnomalyDetector


def load_mask(path, image_size):
    if path is None or not Path(path).exists():
        return None
    mask = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.Resampling.NEAREST
    )
    return np.asarray(mask) > 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MVTec AD test images.")
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--model", default="outputs/model.pt")
    parser.add_argument("--output-csv", default="outputs/evaluation.csv")
    parser.add_argument(
    "--threshold",
    type=float,
    default=0.70,
    help="Demo threshold for classifying anomalous images."
    )
    args = parser.parse_args()

    detector = PatchAnomalyDetector.load(args.model)
    category_dir = find_mvtec_category(args.data_root, args.category)
    samples = get_test_samples(category_dir)
    transform = image_transform(detector.image_size)

    y_true, scores, rows = [], [], []
    pixel_true, pixel_scores = [], []
    print(f"Evaluating {len(samples)} images for category {args.category}...")
    for index, (path, is_anomaly, mask_path) in enumerate(samples, start=1):
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        score, anomaly_map = detector.predict_tensor(tensor)
        score = float(score[0].cpu())
        y_true.append(int(is_anomaly))
        scores.append(score)
        rows.append({
            "image": str(path),
            "is_anomaly": int(is_anomaly),
            "anomaly_score": score,
            "mask": str(mask_path) if mask_path else "",
        })

        mask = load_mask(mask_path, detector.image_size)
        if mask is not None:
            # Evaluate pixel scores on a resized grid to keep memory manageable.
            pixel_true.extend(mask.flatten().astype(np.uint8).tolist())
            pixel_scores.extend(anomaly_map[0].detach().cpu().numpy().flatten().tolist())
        if index % 25 == 0 or index == len(samples):
            print(f"Processed {index}/{len(samples)}")

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) == 2:
        image_auc = roc_auc_score(y_true, scores)
        print(f"Image-level AUROC: {image_auc:.4f}")
    else:
        image_auc = float("nan")
        print("Image-level AUROC unavailable: test data contains only one label.")

    # AUROC does not require a fixed threshold. Threshold-dependent results are
    # intentionally not printed because a validation-derived threshold is needed.
    if pixel_true and len(np.unique(pixel_true)) == 2:
        pixel_auc = roc_auc_score(pixel_true, np.asarray(pixel_scores))
        print(f"Pixel-level AUROC (resized masks): {pixel_auc:.4f}")
    else:
        pixel_auc = float("nan")
        print("Pixel-level AUROC unavailable (masks missing or one pixel label only).")
    
    # Threshold-based image classification metrics.
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    y_pred = (scores >= args.threshold).astype(int)

    print(f"\nClassification metrics at threshold {args.threshold:.2f}")
    print(f"Accuracy: {(y_pred == y_true).mean():.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted Normal  Predicted Anomaly")
    print(f"Actual Normal       {cm[0, 0]:>5}              {cm[0, 1]:>5}")
    print(f"Actual Anomaly      {cm[1, 0]:>5}              {cm[1, 1]:>5}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-image results saved to {output_path}")


if __name__ == "__main__":
    main()
