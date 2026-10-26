# Industrial Anomaly Detection — College Project

A practical visual-inspection demo that learns normal product appearance and flags unusual regions in new images. It is designed for Google Colab and a one-day college project.

## Real-world use case

A manufacturing camera captures each product. The system compares the image with defect-free examples and returns:
- an image-level anomaly score,
- a heatmap of suspicious regions,
- an overlay for human review.

Potential applications include surface inspection of bottles, tiles, metal components, and similar products. This is a research prototype, not a certified factory inspection system.

## Method

This repository implements a **PatchCore-inspired** method:
1. A pretrained ResNet18 extracts features at two intermediate layers.
2. Feature vectors from defect-free training images form a memory bank.
3. At inference, each test-image patch is compared with its nearest normal feature.
4. Large nearest-neighbour distances produce a higher anomaly score and heatmap.

This is a lightweight educational implementation, **not the official PatchCore implementation**. Its scores and metrics may differ from published PatchCore results.

## Dataset

Use the official MVTec AD dataset:
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Download it from the official page and review the license before use. It is intended for non-commercial research use under its published license; do not use or redistribute it commercially without permission.

Start with one category, such as `bottle`. The expected structure is:

```text
data/mvtec_anomaly_detection/
└── bottle/
    ├── train/
    │   └── good/
    ├── test/
    │   ├── good/
    │   └── <defect_type>/
    └── ground_truth/
        └── <defect_type>/
```

The dataset archive may extract into an extra nested directory. Set `--data-root` to the directory containing the category folder. The code attempts to recognize common nested layouts.

## Option A — Run in Google Colab (recommended)

1. Upload `industrial_anomaly_detection_project.zip` to Colab, or upload it to Google Drive and mount Drive.
2. Extract the ZIP:

```python
!unzip -q industrial_anomaly_detection_project.zip -d /content/
%cd /content/industrial_anomaly_detection_project
```

If you uploaded/extracted it under a different path, change `%cd` to the actual folder.

3. Install requirements:

```python
!pip install -r requirements.txt
```

4. Check whether a GPU is available:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

In Colab, choose **Runtime → Change runtime type → T4 GPU** if one is available. A GPU is recommended, but availability varies.

5. Download the official MVTec AD archive from the dataset page. Upload the extracted dataset into the Colab session or Drive. For example, the project expects the category at:

```text
/content/industrial_anomaly_detection_project/data/mvtec_anomaly_detection/bottle/
```

You can also leave the dataset in Drive and pass that directory to `--data-root`.

6. Verify the dataset path:

```python
from pathlib import Path
root = Path("data/mvtec_anomaly_detection")
print("Root exists:", root.exists())
print("Bottle train folder exists:", (root / "bottle" / "train" / "good").exists())
print("Number of normal training images:",
      len(list((root / "bottle" / "train" / "good").glob("*"))))
```

7. Train the model on normal images only:

```python
!python train.py \
  --data-root data/mvtec_anomaly_detection \
  --category bottle \
  --image-size 256 \
  --batch-size 16 \
  --max-memory-patches 12000 \
  --output outputs/model.pt
```

The first run downloads pretrained ImageNet ResNet18 weights. Keep the runtime connected until training completes. If GPU memory is limited, lower `--batch-size` to `8`; if training is too slow, lower `--image-size` to `224` or `192`.

8. Evaluate on the MVTec test split:

```python
!python evaluate.py \
  --data-root data/mvtec_anomaly_detection \
  --category bottle \
  --model outputs/model.pt \
  --output-csv outputs/evaluation.csv
```

The script reports image-level AUROC and pixel-level AUROC when labels/masks permit, and saves per-image scores to `outputs/evaluation.csv`. These metrics are for a quick prototype; the pixel AUROC uses resized masks and should be reported with that limitation.

9. Launch the web demo:

```python
!python app.py --model outputs/model.pt --share
```

Open the temporary Gradio link shown in the output. Upload normal and defective test images. The UI displays the original/heatmap overlay, heatmap, and anomaly score.

10. Keep the notebook/runtime active while using a temporary Gradio share link. Do not upload confidential factory images to a public demo link.

## Option B — Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train.py --data-root data/mvtec_anomaly_detection --category bottle
python evaluate.py --data-root data/mvtec_anomaly_detection --category bottle
python app.py --model outputs/model.pt
```

On Windows, run the activation command appropriate to your terminal. For a local demo, omit `--share`.

## Project layout

```text
industrial_anomaly_detection_project/
├── app.py              # Gradio upload demo
├── train.py            # builds the normal feature memory bank
├── evaluate.py         # test metrics and per-image scores
├── requirements.txt
├── README.md
├── src/
│   ├── data.py         # dataset paths and image transforms
│   └── model.py        # feature extraction, memory bank, anomaly maps
├── data/               # put the dataset here (not included)
└── outputs/             # model and evaluation output (created at runtime)
```

## Troubleshooting

### `Could not find the normal training folder`
Check the directory path. It must contain `bottle/train/good` (or the selected category's `train/good`). Use:

```python
!find data -maxdepth 4 -type d | head -80
```

Then pass the correct parent directory to `--data-root`.

### `No module named torch` or other missing package
Run `pip install -r requirements.txt`, then restart the Colab session if imports still fail.

### CUDA out-of-memory
Try `--batch-size 8`, then `4`. Reduce image size to `224` or `192` if needed.

### Very slow inference
Nearest-neighbour distance grows with memory-bank size. Reduce `--max-memory-patches` to `6000` or `3000` when training, then retrain. This trades some detection detail for speed.

### The heatmap is noisy or highlights the background
This can happen due to changes in lighting, image alignment, or background. Use consistent product images, compare multiple examples, and document failures. Do not claim that every highlighted region is a true defect.

### Everything appears anomalous
The model is not using a universal calibrated threshold. Compare scores on known normal and defective test images; do not interpret the numeric score as a probability. Ensure the selected category matches the training data.

### Model checkpoint does not exist
Run the training command successfully first. It should create `outputs/model.pt`.

## One-day schedule

- 0:00–0:45 — obtain/extract dataset, install dependencies, check paths.
- 0:45–2:00 — train on one category.
- 2:00–3:00 — run evaluation and inspect outputs.
- 3:00–4:00 — launch Gradio demo and test known images.
- 4:00–6:00 — prepare screenshots, metrics, report, and viva explanation.
- Remaining time — debug and add a second category only if the first works.

## What to include in the college report

1. Problem statement and industrial use case.
2. Dataset description and license.
3. Method: pretrained feature extraction, memory bank, nearest-neighbour patch scoring.
4. Evaluation: image-level AUROC, pixel-level AUROC, and example heatmaps.
5. Failure cases and limitations.
6. Demo screenshots and future scope.

## Important limitations

- The model is specialized to the product category used in training.
- A benchmark dataset is not the same as live factory camera data.
- Anomaly score is not a calibrated probability.
- The heatmap indicates unusual appearance, not a verified diagnosis of defect type.
- Use a human review step; do not automatically reject products based on this prototype.
- MVTec AD has its own license terms. Follow them.

## References

- MVTec AD: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- PyTorch: https://pytorch.org/
- Torchvision ResNet: https://pytorch.org/vision/stable/models/resnet.html
- Gradio: https://www.gradio.app/
