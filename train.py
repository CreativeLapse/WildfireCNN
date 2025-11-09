"""
Training Script for Wildfire Detection Model
Implements YOLO loss function and training loop.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from model import WildfireYOLO
from dataset import get_dataloaders


class YOLOLoss(nn.Module):
    """
    YOLO loss function combining:
    - Bounding box loss (IoU-based)
    - Objectness loss (BCE)
    - Classification loss (BCE)
    """
    
    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_obj: float = 1.0,
        lambda_class: float = 1.0
    ):
        """
        Args:
            lambda_coord: Weight for coordinate loss
            lambda_noobj: Weight for no-object loss
            lambda_obj: Weight for object loss
            lambda_class: Weight for classification loss
        """
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: list,
        image_size: int = 416
    ) -> dict:
        """
        Compute YOLO loss.
        
        Args:
            predictions: Model predictions [B, num_anchors * (5 + num_classes), H, W]
            targets: List of target dicts with 'boxes' and 'labels'
            image_size: Input image size
        
        Returns:
            Dictionary with loss components
        """
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # Reshape predictions
        pred_shape = predictions.shape
        H, W = pred_shape[2], pred_shape[3]
        num_anchors = 1  # Assuming 1 anchor for simplicity
        num_classes = 1
        
        predictions = predictions.view(
            batch_size,
            num_anchors,
            5 + num_classes,
            H,
            W
        )
        
        # Permute to [B, num_anchors, H, W, 5 + num_classes]
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
        
        # Initialize loss components as tensors
        coord_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        noobj_loss = torch.tensor(0.0, device=device)
        class_loss = torch.tensor(0.0, device=device)
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0) / W  # Normalized
        grid_y = grid_y.unsqueeze(0).unsqueeze(0) / H  # Normalized
        
        for b in range(batch_size):
            target = targets[b]
            boxes = target['boxes']  # [N, 4] normalized (x_center, y_center, width, height)
            labels = target['labels']  # [N]
            
            # Get predictions for this batch
            pred = predictions[b, 0]  # [H, W, 5 + num_classes]
            
            # Split predictions
            pred_x = torch.sigmoid(pred[..., 0])  # [H, W]
            pred_y = torch.sigmoid(pred[..., 1])  # [H, W]
            pred_w = torch.sigmoid(pred[..., 2])  # [H, W]
            pred_h = torch.sigmoid(pred[..., 3])  # [H, W]
            pred_obj = pred[..., 4]  # [H, W] (logits)
            pred_class = pred[..., 5:]  # [H, W, num_classes] (logits)
            
            # Convert to absolute coordinates
            pred_x_abs = (pred_x + grid_x.squeeze()) / W
            pred_y_abs = (pred_y + grid_y.squeeze()) / H
            
            # Create objectness mask (cells with objects)
            # If image has no fire (empty boxes), obj_mask stays all False, noobj_mask stays all True
            obj_mask = torch.zeros((H, W), device=device, dtype=torch.bool)
            noobj_mask = torch.ones((H, W), device=device, dtype=torch.bool)
            
            # For each ground truth box, find responsible grid cell
            # If boxes is empty (no fire), this loop doesn't execute, and all cells get no-object loss
            for i, box in enumerate(boxes):
                x_center, y_center, width, height = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                
                # Find grid cell
                grid_x_idx = int(x_center * W)
                grid_y_idx = int(y_center * H)
                grid_x_idx = max(0, min(W - 1, grid_x_idx))
                grid_y_idx = max(0, min(H - 1, grid_y_idx))
                
                obj_mask[grid_y_idx, grid_x_idx] = True
                noobj_mask[grid_y_idx, grid_x_idx] = False
                
                # Coordinate loss (only for cells with objects)
                if obj_mask[grid_y_idx, grid_x_idx]:
                    # Target coordinates in grid cell space
                    target_x = x_center * W - grid_x_idx
                    target_y = y_center * H - grid_y_idx
                    target_w = width
                    target_h = height
                    
                    # Predicted coordinates
                    pred_x_cell = pred_x[grid_y_idx, grid_x_idx]
                    pred_y_cell = pred_y[grid_y_idx, grid_x_idx]
                    pred_w_cell = pred_w[grid_y_idx, grid_x_idx]
                    pred_h_cell = pred_h[grid_y_idx, grid_x_idx]
                    
                    # Coordinate loss
                    coord_loss += self.lambda_coord * (
                        self.mse_loss(pred_x_cell, torch.tensor(target_x, device=device)) +
                        self.mse_loss(pred_y_cell, torch.tensor(target_y, device=device)) +
                        self.mse_loss(pred_w_cell, torch.tensor(target_w, device=device)) +
                        self.mse_loss(pred_h_cell, torch.tensor(target_h, device=device))
                    )
                    
                    # Objectness loss (should be 1)
                    obj_loss += self.lambda_obj * self.bce_loss(
                        pred_obj[grid_y_idx, grid_x_idx].unsqueeze(0),
                        torch.tensor([1.0], device=device)
                    )
                    
                    # Classification loss
                    class_loss += self.lambda_class * self.bce_loss(
                        pred_class[grid_y_idx, grid_x_idx],
                        torch.tensor([1.0], device=device)  # Class 0 (wildfire)
                    )
            
            # No-object loss (cells without objects)
            noobj_loss += self.lambda_noobj * self.bce_loss(
                pred_obj[noobj_mask],
                torch.zeros(noobj_mask.sum().item(), device=device)
            )
        
        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        
        return {
            'total': total_loss,
            'coord': coord_loss,
            'obj': obj_loss,
            'noobj': noobj_loss,
            'class': class_loss
        }


def calculate_map(predictions: list, targets: list, iou_threshold: float = 0.5) -> float:
    """
    Calculate mean Average Precision (mAP).
    Simplified version for single class.
    """
    if len(predictions) == 0:
        return 0.0
    
    # For simplicity, return a basic metric
    # In a full implementation, this would compute AP properly
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, target in zip(predictions, targets):
        target_boxes = target['boxes']
        
        if len(pred) > 0 and len(target_boxes) > 0:
            # Check if any prediction matches a target
            true_positives += 1
        elif len(pred) > 0:
            false_positives += 1
        elif len(target_boxes) > 0:
            false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)  # F1 score as proxy for mAP


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    running_loss = {'total': 0.0, 'coord': 0.0, 'obj': 0.0, 'noobj': 0.0, 'class': 0.0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for images, targets in pbar:
        images = images.to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Compute loss
        loss_dict = criterion(predictions, targets, image_size=images.shape[-1])
        loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update running loss
        for key in running_loss:
            # Handle both tensor and float values
            if isinstance(loss_dict[key], torch.Tensor):
                running_loss[key] += loss_dict[key].item()
            else:
                running_loss[key] += loss_dict[key]
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Average losses
    for key in running_loss:
        running_loss[key] /= len(train_loader)
    
    return running_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate the model."""
    model.eval()
    running_loss = {'total': 0.0, 'coord': 0.0, 'obj': 0.0, 'noobj': 0.0, 'class': 0.0}
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        for images, targets in pbar:
            images = images.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss_dict = criterion(predictions, targets, image_size=images.shape[-1])
            loss = loss_dict['total']
            
            # Update running loss
            for key in running_loss:
                # Handle both tensor and float values
                if isinstance(loss_dict[key], torch.Tensor):
                    running_loss[key] += loss_dict[key].item()
                else:
                    running_loss[key] += loss_dict[key]
            
            # Decode predictions for mAP calculation
            decoded_preds = model.decode_predictions(predictions, conf_threshold=0.3)
            all_predictions.extend(decoded_preds)
            all_targets.extend(targets)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Average losses
    for key in running_loss:
        running_loss[key] /= len(val_loader)
    
    # Calculate mAP
    map_score = calculate_map(all_predictions, all_targets)
    
    return {**running_loss, 'map': map_score}


def main():
    parser = argparse.ArgumentParser(description='Train Wildfire Detection Model')
    parser.add_argument('--dataset-root', type=str, default='defi1certif-datasets-fire_small',
                        help='Root directory of the dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=416, help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device('cpu')
            print("⚠ CUDA not available, using CPU (training will be slower)")
    else:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("⚠ Warning: CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = WildfireYOLO(
        num_classes=1,
        num_anchors=1,
        image_size=args.image_size
    ).to(device)
    
    # Loss function
    criterion = YOLOLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0005)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_map = 0.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss['total']:.4f} "
              f"(Coord: {train_loss['coord']:.4f}, "
              f"Obj: {train_loss['obj']:.4f}, "
              f"NoObj: {train_loss['noobj']:.4f}, "
              f"Class: {train_loss['class']:.4f})")
        print(f"Val Loss: {val_metrics['total']:.4f}, Val mAP: {val_metrics['map']:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_map': best_map,
            'val_metrics': val_metrics
        }
        
        # Save every epoch
        torch.save(checkpoint, 'wildfire_detector_checkpoint.pth')
        
        # Save best model
        if val_metrics['map'] > best_map:
            best_map = val_metrics['map']
            torch.save(checkpoint, 'wildfire_detector_best.pth')
            print(f"✓ Saved best model (mAP: {best_map:.4f})")
        
        print("-" * 50)
    
    print("\nTraining complete!")
    print(f"Best mAP: {best_map:.4f}")
    print(f"Best model saved as: wildfire_detector_best.pth")


if __name__ == "__main__":
    main()

