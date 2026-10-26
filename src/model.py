from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from .data import NormalImageDataset


class PatchAnomalyDetector:
    """
    A compact PatchCore-inspired detector.

    It extracts pretrained ResNet18 layer2/layer3 patch features from normal
    training images, stores a representative random subset in a memory bank,
    and scores test patches by nearest-neighbour distance to that bank.

    This is an educational, lightweight implementation, not the official
    PatchCore implementation.
    """
    def __init__(self, image_size=256, device=None, pretrained=True):
        self.image_size = int(image_size)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = backbone.to(self.device).eval()
        self.memory_bank = None
        self.feature_mean = None
        self.feature_std = None

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def extract_features(self, batch):
        x = batch.to(self.device)
        m = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(
            self.backbone.conv1(x)
        )))
        m = self.backbone.layer1(m)
        layer2 = self.backbone.layer2(m)       # roughly 1/8 resolution
        layer3 = self.backbone.layer3(layer2)  # roughly 1/16 resolution
        layer3 = F.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        # Normalize each feature channel spatially only by combining layers;
        # distance is computed in the resulting joint feature space.
        features = torch.cat([layer2, layer3], dim=1)
        features = F.normalize(features, p=2, dim=1)
        # B x C x H x W -> B x H x W x C
        return features.permute(0, 2, 3, 1).contiguous()

    def fit(self, image_paths, batch_size=16, max_memory_patches=12000,
            num_workers=2, seed=42):
        if not image_paths:
            raise ValueError("No normal training images were provided.")
        random.seed(seed)
        np.random.seed(seed)
        dataset = NormalImageDataset(image_paths, self.image_size)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(self.device.type == "cuda")
        )
        collected = []
        for batch, _paths in loader:
            feats = self.extract_features(batch)
            collected.append(feats.reshape(-1, feats.shape[-1]).cpu())
        all_features = torch.cat(collected, dim=0)

        # Random representative subset keeps nearest-neighbour inference manageable.
        if len(all_features) > max_memory_patches:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(all_features), generator=generator)[:max_memory_patches]
            all_features = all_features[indices]

        self.memory_bank = F.normalize(all_features.float(), p=2, dim=1).to(self.device)
        return {
            "training_images": len(image_paths),
            "patches_seen": int(sum(x.shape[0] for x in collected)),
            "memory_bank_size": int(self.memory_bank.shape[0]),
            "device": str(self.device),
        }

    @torch.inference_mode()
    def predict_tensor(self, batch, query_chunk_size=512):
        if self.memory_bank is None:
            raise RuntimeError("Model has no memory bank. Train or load a checkpoint first.")
        features = self.extract_features(batch)
        batch_size, height, width, channels = features.shape
        flat = features.reshape(-1, channels)
        min_distances = []
        # Chunk query patches to limit peak memory during torch.cdist.
        for start in range(0, flat.shape[0], query_chunk_size):
            query = flat[start:start + query_chunk_size]
            distances = torch.cdist(query, self.memory_bank)
            min_distances.append(distances.min(dim=1).values)
        patch_scores = torch.cat(min_distances).reshape(batch_size, height, width)
        anomaly_map = F.interpolate(
            patch_scores.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        # Average the top 1% of anomaly-map values.
        flat_map = anomaly_map.flatten(1)
        k = max(1, int(flat_map.shape[1] * 0.01))
        image_scores = flat_map.topk(k, dim=1).values.mean(dim=1)
        return image_scores, anomaly_map

    def predict_image(self, image: Image.Image):
        from .data import image_transform
        transform = image_transform(self.image_size)
        tensor = transform(image.convert("RGB")).unsqueeze(0)
        score, anomaly_map = self.predict_tensor(tensor)
        return float(score[0].cpu()), anomaly_map[0].cpu().numpy()

    def save(self, path, metadata=None):
        if self.memory_bank is None:
            raise RuntimeError("Train the model before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "image_size": self.image_size,
            "backbone_state_dict": self.backbone.state_dict(),
            "memory_bank": self.memory_bank.detach().cpu(),
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # No pretrained download is needed: checkpoint contains backbone weights.
        detector = cls(
            image_size=checkpoint["image_size"],
            device=device,
            pretrained=False,
        )
        detector.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        detector.memory_bank = checkpoint["memory_bank"].float().to(detector.device)
        detector.metadata = checkpoint.get("metadata", {})
        return detector
