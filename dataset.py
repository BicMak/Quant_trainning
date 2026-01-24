"""
ImageNet-Mini Dataset for PyTorch
Supports train/val splits with standard ImageFolder structure
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageNetMiniDataset(Dataset):
    """
    ImageNet-Mini Dataset

    Expected directory structure:
        imagenet-mini/
            train/
                n01440764/
                    n01440764_10043.JPEG
                    n01440764_10470.JPEG
                    ...
                n01443537/
                    ...
            val/
                n01440764/
                    ...

    Args:
        root_dir: Path to imagenet-mini directory
        split: 'train' or 'val'
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to labels
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        assert split in ['train', 'val'], f"split must be 'train' or 'val', got {split}"

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Get split directory
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Build class to index mapping
        self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # Build samples list (image_path, label)
        self.samples = []
        self._load_samples()

        print(f"Loaded {len(self.samples)} images from {split} split ({len(self.classes)} classes)")

    def _load_samples(self):
        """Load all image paths and labels"""
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            class_idx = self.class_to_idx[class_name]

            # Get all image files in this class directory
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    self.samples.append((str(img_path), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at index

        Returns:
            image: Transformed image tensor
            label: Class index
        """
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """Get class name from index"""
        return self.idx_to_class.get(idx, "Unknown")

    def get_num_classes(self) -> int:
        """Get total number of classes"""
        return len(self.classes)


def get_imagenet_transforms(split: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    Get standard ImageNet transforms for ViT

    Args:
        split: 'train' or 'val'
        img_size: Target image size (default 224 for ViT)

    Returns:
        Composed transforms
    """
    if split == 'train':
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(
    root_dir: str = 'imagenet-mini',
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    shuffle_train: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        root_dir: Path to imagenet-mini directory
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        img_size: Target image size
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = ImageNetMiniDataset(
        root_dir=root_dir,
        split='train',
        transform=get_imagenet_transforms('train', img_size)
    )

    val_dataset = ImageNetMiniDataset(
        root_dir=root_dir,
        split='val',
        transform=get_imagenet_transforms('val', img_size)
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def test_dataset():
    """Test dataset loading and visualization"""
    import matplotlib.pyplot as plt
    import numpy as np

    print("="*80)
    print("Testing ImageNet-Mini Dataset")
    print("="*80)

    # Create dataset
    dataset = ImageNetMiniDataset(
        root_dir='imagenet-mini',
        split='train',
        transform=get_imagenet_transforms('train', img_size=224)
    )

    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(dataset)}")
    print(f"  Number of classes: {dataset.get_num_classes()}")
    print(f"  Classes (first 10): {dataset.classes[:10]}")

    # Test single sample
    print(f"\n[Single Sample Test]")
    img, label = dataset[0]
    print(f"  Image shape: {img.shape}")
    print(f"  Label: {label}")
    print(f"  Class name: {dataset.get_class_name(label)}")

    # Test dataloader
    print(f"\n[DataLoader Test]")
    train_loader, val_loader = create_dataloaders(
        root_dir='imagenet-mini',
        batch_size=8,
        num_workers=2,
        img_size=224
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Get one batch
    images, labels = next(iter(train_loader))
    print(f"\n[Batch Sample]")
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Batch labels: {labels.tolist()}")

    # Visualize first batch
    print(f"\n[Visualization]")

    def denormalize(tensor):
        """Denormalize image for visualization"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx in range(min(8, len(images))):
        img = denormalize(images[idx])
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img)
        axes[idx].set_title(f"Label: {labels[idx].item()}\n{dataset.get_class_name(labels[idx].item())}")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_samples.jpg', dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: dataset_samples.jpg")
    plt.close()

    print("\n" + "="*80)
    print("Dataset test completed successfully!")
    print("="*80)


if __name__ == '__main__':
    test_dataset()