"""
Custom PyTorch Dataset for Wildfire Detection
Handles image loading, annotation parsing, and data augmentation.
Supports YOLO format annotations.
"""
import os
import glob
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class WildfireDataset(Dataset):
    """
    Custom Dataset for wildfire detection with YOLO format annotations.
    
    Supports two dataset formats:
    1. YOLO format (data_bb): train/images, train/labels with .txt files
    2. Old format: fire/, no_fire/, start_fire/ folders (for backward compatibility)
    """
    
    def __init__(
        self,
        dataset_root: str,
        image_size: int = 416,
        augment: bool = True,
        split: str = "train",
        train_ratio: float = 0.8
    ):
        """
        Args:
            dataset_root: Root directory of the dataset
            image_size: Target image size (square)
            augment: Whether to apply data augmentation
            split: "train", "val", or "test"
            train_ratio: Ratio of data to use for training (only used if split is train/val)
        """
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.augment = augment and (split == "train")
        
        # Collect all images
        self.image_paths = []
        self.annotations = []  # List of lists: each inner list contains [class_id, x_center, y_center, width, height] (normalized)
        
        # Check if this is YOLO format dataset (has train/images, train/labels structure)
        train_images_dir = self.dataset_root / "train" / "images"
        train_labels_dir = self.dataset_root / "train" / "labels"
        valid_images_dir = self.dataset_root / "valid" / "images"
        valid_labels_dir = self.dataset_root / "valid" / "labels"
        test_images_dir = self.dataset_root / "test" / "images"
        test_labels_dir = self.dataset_root / "test" / "labels"
        
        if train_images_dir.exists() and train_labels_dir.exists():
            # YOLO format dataset
            print(f"Loading YOLO format dataset from {dataset_root}")
            
            if split == "train":
                images_dir = train_images_dir
                labels_dir = train_labels_dir
            elif split == "val":
                images_dir = valid_images_dir
                labels_dir = valid_labels_dir
            elif split == "test":
                images_dir = test_images_dir
                labels_dir = test_labels_dir
            else:
                raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")
            
            # Load all images
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            for img_path in image_files:
                # Find corresponding label file
                label_path = labels_dir / (img_path.stem + ".txt")
                
                self.image_paths.append(str(img_path))
                
                # Load annotations from YOLO format file
                if label_path.exists() and label_path.stat().st_size > 0:
                    annotations = []
                    with open(label_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    # Only use class 0 (fire) for now, ignore smoke (class 1)
                                    if class_id == 0:
                                        annotations.append([class_id, x_center, y_center, width, height])
                    self.annotations.append(annotations)
                else:
                    # No annotations (negative sample)
                    self.annotations.append([])
        
        else:
            # Old format: fire/, no_fire/, start_fire/ folders
            print(f"Loading old format dataset from {dataset_root}")
            
            # Load fire images (positive samples with full-image bounding boxes)
            fire_dir = self.dataset_root / "fire"
            start_fire_dir = self.dataset_root / "start_fire"
            
            if fire_dir.exists():
                fire_images = list(fire_dir.glob("*.jpg"))
                for img_path in fire_images:
                    self.image_paths.append(str(img_path))
                    # Full-image bounding box: center at (0.5, 0.5), size (1.0, 1.0)
                    self.annotations.append([[0, 0.5, 0.5, 1.0, 1.0]])
            
            if start_fire_dir.exists():
                start_fire_images = list(start_fire_dir.glob("*.jpg"))
                for img_path in start_fire_images:
                    self.image_paths.append(str(img_path))
                    # Full-image bounding box
                    self.annotations.append([[0, 0.5, 0.5, 1.0, 1.0]])
            
            # Load no_fire images (negative samples with no bounding boxes)
            no_fire_dir = self.dataset_root / "no_fire"
            if no_fire_dir.exists():
                no_fire_images = list(no_fire_dir.glob("*.jpg"))
                for img_path in no_fire_images:
                    self.image_paths.append(str(img_path))
                    # No bounding boxes (empty list)
                    self.annotations.append([])
            
            # Shuffle and split (only for old format)
            combined = list(zip(self.image_paths, self.annotations))
            random.seed(42)
            random.shuffle(combined)
            
            split_idx = int(len(combined) * train_ratio)
            if split == "train":
                combined = combined[:split_idx]
            else:
                combined = combined[split_idx:]
            
            self.image_paths, self.annotations = zip(*combined) if combined else ([], [])
            self.image_paths = list(self.image_paths)
            self.annotations = list(self.annotations)
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
        print(f"  - Positive samples (with fire): {sum(len(ann) > 0 for ann in self.annotations)}")
        print(f"  - Negative samples (no fire): {sum(len(ann) == 0 for ann in self.annotations)}")
    
    def _setup_transforms(self):
        """Setup image transforms for training/validation."""
        if self.augment:
            # Training transforms with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation transforms (no augmentation)
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: Tensor of shape [3, H, W]
            target: Dict with keys:
                - 'boxes': Tensor of shape [N, 4] (x_center, y_center, width, height, normalized)
                - 'labels': Tensor of shape [N] (class_ids)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Get annotations
        anns = self.annotations[idx]
        
        # Apply horizontal flip augmentation if needed
        should_flip = self.augment and random.random() < 0.5
        if should_flip and anns:
            image = transforms.functional.hflip(image)
            # Flip x_center coordinates
            for ann in anns:
                ann[1] = 1.0 - ann[1]  # x_center = 1 - x_center
        
        # Apply transforms
        image = self.transform(image)
        
        # Prepare target
        if len(anns) > 0:
            boxes = torch.tensor([[ann[1], ann[2], ann[3], ann[4]] for ann in anns], dtype=torch.float32)
            labels = torch.tensor([ann[0] for ann in anns], dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for DataLoader.
    Handles variable number of bounding boxes per image.
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    return images, targets


def get_dataloaders(
    dataset_root: str,
    batch_size: int = 16,
    image_size: int = 416,
    num_workers: int = 4,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = WildfireDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        augment=True,
        split="train",
        train_ratio=train_ratio
    )
    
    val_dataset = WildfireDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        augment=False,
        split="val",
        train_ratio=train_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    dataset_root = "data_bb"
    
    if os.path.exists(dataset_root):
        train_loader, val_loader = get_dataloaders(
            dataset_root=dataset_root,
            batch_size=4,
            image_size=416,
            num_workers=0
        )
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test a batch
        images, targets = next(iter(train_loader))
        print(f"\nBatch shape: {images.shape}")
        print(f"Number of targets: {len(targets)}")
        if len(targets) > 0:
            print(f"First target boxes shape: {targets[0]['boxes'].shape}")
            print(f"First target labels: {targets[0]['labels']}")
            if len(targets[0]['boxes']) > 0:
                print(f"First box: {targets[0]['boxes'][0]}")
    else:
        print(f"Dataset directory {dataset_root} not found.")
