from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_transform(image_size: int = 256):
    # ImageNet normalization matches the pretrained ResNet18 feature extractor.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


class NormalImageDataset(Dataset):
    def __init__(self, paths, image_size=256):
        self.paths = list(paths)
        self.transform = image_transform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


def find_mvtec_category(root: str, category: str) -> Path:
    """
    Accept either:
      root/category/train/good
    or:
      root/mvtec_anomaly_detection/category/train/good
    or root already pointing at the category folder.
    """
    root_path = Path(root).expanduser().resolve()
    candidates = [
        root_path / category,
        root_path / "mvtec_anomaly_detection" / category,
        root_path / "MVTecAD" / category,
        root_path,
    ]
    for candidate in candidates:
        if (candidate / "train" / "good").is_dir():
            return candidate
    expected = "\n".join(str(c / "train" / "good") for c in candidates)
    raise FileNotFoundError(
        f"Could not find the normal training folder for category '{category}'. "
        f"Expected train/good under one of:\n{expected}\n"
        "Set --data-root to the directory that contains the category folder."
    )


def get_training_paths(category_dir: Path):
    paths = list_images(category_dir / "train" / "good")
    if not paths:
        raise FileNotFoundError(
            f"No training images found in {category_dir / 'train' / 'good'}"
        )
    return paths


def get_test_samples(category_dir: Path):
    """
    Returns (image_path, is_anomaly, mask_path_or_None).
    MVTec test/good is normal. Other test subfolders are anomalous.
    Masks for anomalous samples are expected under ground_truth/<defect_type>.
    """
    test_dir = category_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    samples = []
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        is_anomaly = defect_dir.name.lower() != "good"
        for image_path in list_images(defect_dir):
            mask_path = None
            if is_anomaly:
                mask_dir = category_dir / "ground_truth" / defect_dir.name
                # MVTec masks usually share image stem and use _mask.png.
                possible = [
                    mask_dir / f"{image_path.stem}_mask.png",
                    mask_dir / f"{image_path.stem}.png",
                ]
                mask_path = next((p for p in possible if p.exists()), None)
            samples.append((image_path, is_anomaly, mask_path))
    if not samples:
        raise FileNotFoundError(f"No test images found in {test_dir}")
    return samples
