# Industrial Anomaly Detection

A visual inspection system that learns the appearance of normal products from defect-free images and flags anomalous regions in new samples. Built on a PatchCore-inspired approach, this project provides a complete pipeline for training, evaluation, and interactive demo.

## Overview

In manufacturing quality control, automated visual inspection can help detect surface defects such as scratches, dents, stains, or cracks. This project tackles the problem with **one-class classification**: the model learns what a normal product looks like using only defect-free training images. At test time, it identifies regions that deviate from the learned normal appearance.

### Key features

- **Anomaly scoring** — each image receives a numerical score indicating how anomalous it is.
- **Pixel-level heatmaps** — the model highlights specific regions that contribute to the anomaly score.
- **Interactive web demo** — a Gradio-based interface for uploading images and inspecting results.
- **Standard metrics** — reports image-level and pixel-level AUROC on benchmark datasets (e.g. MVTec AD).

## How it works

This project implements a **PatchCore-inspired** method:

1. **Feature extraction** — A pretrained ResNet18 extracts convolutional feature maps at two intermediate layers (layer2 and layer3) from each training image. These features encode local visual patterns at multiple scales.
2. **Memory bank** — The patch-level feature vectors from all defect-free training images are collected, and a random representative subset is stored as a memory bank. This bank serves as the reference for "normal" appearance.
3. **Nearest-neighbour scoring** — At inference, each patch of a test image is compared (via Euclidean distance in feature space) to its nearest neighbour in the memory bank. Patches that look unlike any normal training patch receive a high distance score.
4. **Anomaly map** — The per-patch distances are upsampled to the original image resolution, producing a continuous heatmap. The score for the entire image is the average of the top 1% of heatmap values.

> This is a lightweight educational implementation inspired by the PatchCore paper. It is not the official PatchCore implementation, and its scores/metrics may differ from published results.

## Dataset

The system is designed for use with the **MVTec AD** dataset, a benchmark for anomaly detection in industrial images:

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

The dataset is intended for non-commercial research use under its published license. Download it from the official page and review the license before use.

Start with one category (e.g. `bottle`). The expected directory structure is:

```text
data/mvtec_anomaly_detection/
└── bottle/
    ├── train/
    │   └── good/          ← defect-free training images
    ├── test/
    │   ├── good/          ← normal test images
    │   └── <defect_type>/ ← defective test images (e.g. broken_large, contamination)
    └── ground_truth/
        └── <defect_type>/ ← pixel-level mask images
```

The dataset archive may extract into an extra nested directory. Use `--data-root` to point to the directory containing the category folder; the code recognises common nested layouts.

## Quick start

### Requirements

- Python 3.10 or 3.11 recommended
- PyTorch (see [pytorch.org](https://pytorch.org/) for platform-specific installation)
- See `requirements.txt` for additional packages

### 1. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train

Train the model on normal images of a single category:

```bash
python train.py \
  --data-root data/mvtec_anomaly_detection \
  --category bottle \
  --image-size 256 \
  --batch-size 16 \
  --max-memory-patches 12000 \
  --output outputs/model.pt
```

The first run downloads pretrained ResNet18 weights from ImageNet. If GPU memory is limited, reduce `--batch-size` to 8 or `--image-size` to 224.

### 3. Evaluate

Run evaluation on the MVTec test split:

```bash
python evaluate.py \
  --data-root data/mvtec_anomaly_detection \
  --category bottle \
  --model outputs/model.pt \
  --output-csv outputs/evaluation.csv
```

The script reports image-level AUROC and pixel-level AUROC (when ground-truth masks are available) and saves per-image scores to a CSV.

### 4. Launch the web interface

```bash
python app.py --model outputs/model.pt
```

Open the local Gradio link in a browser. Upload normal and defective test images to see:
- The original image overlaid with the anomaly heatmap
- The raw heatmap
- The anomaly score and inspection decision

Add `--share` to create a temporary public link for sharing the demo.

## Project structure

```text
industrial_anomaly_detection_project/
├── app.py              # Gradio web demo
├── train.py            # builds the normal-feature memory bank
├── evaluate.py         # test metrics and per-image scores
├── requirements.txt
├── README.md
├── src/
│   ├── data.py         # dataset paths, image transforms, helpers
│   └── model.py        # feature extraction, memory bank, anomaly scoring
├── data/               # place the dataset here (not committed)
└── outputs/            # model checkpoints and evaluation results (created at runtime)
```

## Configuration tips

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--image-size` | 256 | Lower to 224 or 192 if GPU memory is limited or training is slow |
| `--batch-size` | 16 | Reduce to 8 or 4 for low-memory GPUs |
| `--max-memory-patches` | 12000 | Larger = more representative memory bank but slower inference; smaller = faster but may miss subtle defects |
| `--threshold` (in app.py) | 0.55 | Demo decision threshold only; validate against your own data before any real use |

## Limitations

- The model is specialised to the product category it was trained on. A bottle model cannot inspect tiles.
- Benchmark datasets (e.g. MVTec AD) do not fully represent real factory-camera conditions (lighting, alignment, background variation).
- The anomaly score is not a calibrated probability. It measures dissimilarity from the training set, not a verified defect type.
- The heatmap indicates regions that appear unusual relative to the normal training data; not every highlighted region corresponds to a true defect.
- This is a research prototype. **Always include a human review step.** Do not use it as an autonomous quality-control decision system.

## References

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) — Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019.
- [PatchCore](https://github.com/amazon-science/patchcore-inspection) — Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
- [PyTorch](https://pytorch.org/)
- [Torchvision ResNet](https://pytorch.org/vision/stable/models/resnet.html)
- [Gradio](https://www.gradio.app/)
