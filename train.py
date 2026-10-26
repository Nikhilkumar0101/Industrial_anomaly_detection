import argparse
import json
from pathlib import Path
from src.data import find_mvtec_category, get_training_paths
from src.model import PatchAnomalyDetector


def main():
    parser = argparse.ArgumentParser(
        description="Train a PatchCore-inspired anomaly detector on MVTec AD normal images."
    )
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection",
                        help="Directory containing category folders.")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-memory-patches", type=int, default=12000)
    parser.add_argument("--output", default="outputs/model.pt")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    category_dir = find_mvtec_category(args.data_root, args.category)
    train_paths = get_training_paths(category_dir)
    print(f"Category directory: {category_dir}")
    print(f"Normal training images: {len(train_paths)}")

    detector = PatchAnomalyDetector(image_size=args.image_size)
    summary = detector.fit(
        train_paths,
        batch_size=args.batch_size,
        max_memory_patches=args.max_memory_patches,
        num_workers=args.workers,
    )
    metadata = {
        "category": args.category,
        "category_dir": str(category_dir),
        "training_summary": summary,
        "method": "PatchCore-inspired ResNet18 patch nearest-neighbour",
    }
    detector.save(args.output, metadata=metadata)
    Path(args.output).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"\nSaved model to: {args.output}")


if __name__ == "__main__":
    main()
