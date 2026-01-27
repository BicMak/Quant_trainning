"""
Custom ImageNet dataset with normalization for ViT models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List
import os


class CustomImageNetDataset(Dataset):
    """
    Custom ImageNet dataset class with proper normalization.

    Args:
        root_dir: Path to ImageNet directory (e.g., 'imagenet-mini/train')
        transform: Optional transform to apply to images
        img_size: Image size for resize/crop (default: 224 for ViT)
        max_samples: Maximum number of samples to load (None = all)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
        max_samples: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root_dir}")

        # Default transform with ImageNet normalization
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean (RGB)
                    std=[0.229, 0.224, 0.225]     # ImageNet std (RGB)
                )
            ])
        else:
            self.transform = transform

        # Load image paths and labels
        self.samples = []
        self.class_to_idx = {}
        self._load_dataset(max_samples)

    def _load_dataset(self, max_samples: Optional[int] = None):
        """Load all image paths and create class mapping."""
        # Get all class directories
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        # Create class to index mapping
        self.class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}

        # Load image paths
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            # Get all image files
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), class_idx))

        # Optionally limit samples
        if max_samples is not None and len(self.samples) > max_samples:
            indices = torch.randperm(len(self.samples))[:max_samples].tolist()
            self.samples = [self.samples[i] for i in indices]

        print(f"Loaded {len(self.samples)} images from {len(self.class_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at index.

        Returns:
            Tuple of (transformed_image, class_label)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_imagenet_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get ImageNet transforms with standard normalization.

    ImageNet normalization (RGB):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def get_imagenet_loader(
    data_path: str,
    batch_size: int = 32,
    num_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    img_size: int = 224
) -> DataLoader:
    """
    Create ImageNet data loader using CustomImageNetDataset.

    Args:
        data_path: Path to ImageNet dataset (e.g., 'imagenet-mini/train')
        batch_size: Batch size for data loader
        num_samples: If set, randomly select this many samples from dataset
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        img_size: Image size for resize/crop (default: 224 for ViT)

    Returns:
        DataLoader object
    """
    # Create custom dataset
    dataset = CustomImageNetDataset(
        root_dir=data_path,
        transform=get_imagenet_transforms(img_size=img_size),
        img_size=img_size,
        max_samples=num_samples
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader


def extract_vit_features(
    model,
    data_loader: DataLoader,
    num_batches: Optional[int] = None,
    device: str = 'cpu'
) -> List[torch.Tensor]:
    """
    Extract features from ViT model (input to first attention block).

    For timm ViT models, the pipeline is:
        image -> patch_embed -> cls_token concat -> pos_embed -> pos_drop -> blocks[0]

    Args:
        model: timm ViT model
        data_loader: DataLoader with ImageNet data
        num_batches: Number of batches to extract (None = all batches)
        device: Device to use ('cpu' or 'cuda')

    Returns:
        List of feature tensors
    """
    features = []
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for idx, (images, _) in enumerate(data_loader):
            if num_batches is not None and idx >= num_batches:
                break

            images = images.to(device)

            # Extract patch embeddings
            x = model.patch_embed(images)

            # Add cls token
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

            # Add positional embedding and dropout
            x = model.pos_drop(x + model.pos_embed)

            features.append(x.cpu())

            if (idx + 1) % 10 == 0:
                print(f"  Extracted {idx + 1} batches", end='\r')

    if len(features) > 0:
        print(f"  Extracted {len(features)} batches total")

    return features
