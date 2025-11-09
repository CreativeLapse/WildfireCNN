"""
Test script to evaluate the model on test images.
Helps debug model performance and inference issues.
"""
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import random
from pathlib import Path

from model import WildfireYOLO


def preprocess_image(image_path: str, image_size: int = 416) -> torch.Tensor:
    """Preprocess an image for model input."""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor


def test_model_on_images(
    model_path: str,
    dataset_root: str,
    image_size: int = 416,
    conf_threshold: float = 0.3,
    num_samples: int = 10
):
    """Test the model on sample images from the dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = WildfireYOLO(num_classes=1, num_anchors=1, image_size=image_size).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    dataset_path = Path(dataset_root)
    
    # Check if this is YOLO format dataset (has valid/images, valid/labels structure)
    valid_images_dir = dataset_path / "valid" / "images"
    valid_labels_dir = dataset_path / "valid" / "labels"
    
    if valid_images_dir.exists() and valid_labels_dir.exists():
        # YOLO format dataset - test on validation set
        print("\n" + "="*60)
        print("Testing on VALIDATION set (YOLO format)")
        print("="*60)
        
        # Get all images
        all_images = list(valid_images_dir.glob("*.jpg")) + list(valid_images_dir.glob("*.png"))
        random.seed(42)
        random.shuffle(all_images)
        test_images_list = all_images[:num_samples * 2]  # Get more samples
        
        # Separate into fire and no-fire
        fire_images = []
        no_fire_images = []
        
        for img_path in test_images_list:
            label_path = valid_labels_dir / (img_path.stem + ".txt")
            if label_path.exists() and label_path.stat().st_size > 0:
                # Has annotations - check if it has fire (class 0)
                with open(label_path, 'r') as f:
                    has_fire = False
                    for line in f:
                        if line.strip() and int(line.strip().split()[0]) == 0:
                            has_fire = True
                            break
                    if has_fire:
                        fire_images.append(img_path)
            else:
                # No annotations - no fire
                no_fire_images.append(img_path)
        
        # Test on fire images
        if fire_images:
            print(f"\nTesting on {min(len(fire_images), num_samples)} FIRE images (should detect wildfire)")
            test_images(model, fire_images[:num_samples], device, conf_threshold, "FIRE")
        
        # Test on no_fire images
        if no_fire_images:
            print(f"\nTesting on {min(len(no_fire_images), num_samples)} NO_FIRE images (should NOT detect wildfire)")
            test_images(model, no_fire_images[:num_samples], device, conf_threshold, "NO_FIRE")
    
    else:
        # Old format: fire/, no_fire/, start_fire/ folders
        # Test on fire images
        print("\n" + "="*60)
        print("Testing on FIRE images (should detect wildfire)")
        print("="*60)
        
        fire_dir = dataset_path / "fire"
        if fire_dir.exists():
            fire_images = list(fire_dir.glob("*.jpg"))[:num_samples]
            test_images(model, fire_images, device, conf_threshold, "FIRE")
        
        # Test on no_fire images
        print("\n" + "="*60)
        print("Testing on NO_FIRE images (should NOT detect wildfire)")
        print("="*60)
        
        no_fire_dir = dataset_path / "no_fire"
        if no_fire_dir.exists():
            no_fire_images = list(no_fire_dir.glob("*.jpg"))[:num_samples]
            test_images(model, no_fire_images, device, conf_threshold, "NO_FIRE")
        
        # Test on start_fire images
        print("\n" + "="*60)
        print("Testing on START_FIRE images (should detect wildfire)")
        print("="*60)
        
        start_fire_dir = dataset_path / "start_fire"
        if start_fire_dir.exists():
            start_fire_images = list(start_fire_dir.glob("*.jpg"))[:num_samples]
            test_images(model, start_fire_images, device, conf_threshold, "START_FIRE")


def test_images(model, image_paths, device, conf_threshold, label):
    """Test model on a list of images."""
    correct = 0
    total = len(image_paths)
    
    for img_path in image_paths:
        # Preprocess
        input_tensor = preprocess_image(str(img_path)).to(device)
        
        # Inference
        with torch.no_grad():
            predictions = model(input_tensor)
            detections = model.decode_predictions(
                predictions,
                conf_threshold=conf_threshold,
                nms_threshold=0.4
            )
        
        # Check results
        num_detections = len(detections[0]) if len(detections) > 0 else 0
        
        # For fire images, we expect detections
        # For no_fire images, we expect no detections
        if label == "NO_FIRE":
            expected = 0
            is_correct = num_detections == 0
        else:  # FIRE or START_FIRE
            expected = ">0"
            is_correct = num_detections > 0
        
        if is_correct:
            correct += 1
        
        # Print detailed info
        status = "✓" if is_correct else "✗"
        print(f"{status} {img_path.name}: {num_detections} detection(s)", end="")
        
        if num_detections > 0:
            # Show confidence scores
            confidences = [det[4].item() for det in detections[0]]
            max_conf = max(confidences)
            avg_conf = sum(confidences) / len(confidences)
            print(f" | Max conf: {max_conf:.3f}, Avg conf: {avg_conf:.3f}")
            
            # Show box sizes
            for det in detections[0]:
                x_min, y_min, x_max, y_max = det[0].item(), det[1].item(), det[2].item(), det[3].item()
                box_width = x_max - x_min
                box_height = y_max - y_min
                print(f"    Box: [{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}] "
                      f"Size: {box_width:.2f} x {box_height:.2f}")
        else:
            print()
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n{label} Accuracy: {correct}/{total} ({accuracy:.1f}%)")


def inspect_raw_predictions(model_path: str, image_path: str, image_size: int = 416):
    """Inspect raw model predictions for debugging."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = WildfireYOLO(num_classes=1, num_anchors=1, image_size=image_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Preprocess
    input_tensor = preprocess_image(image_path).to(device)
    
    # Get raw predictions
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Decode
    detections = model.decode_predictions(predictions, conf_threshold=0.1, nms_threshold=0.4)
    
    print(f"\nRaw predictions shape: {predictions.shape}")
    print(f"Number of detections (conf > 0.1): {len(detections[0])}")
    
    # Show statistics
    pred_shape = predictions.shape
    H, W = pred_shape[2], pred_shape[3]
    
    predictions_reshaped = predictions.view(1, 1, 7, H, W).permute(0, 1, 3, 4, 2)
    pred = predictions_reshaped[0, 0]  # [H, W, 7]
    
    objectness = torch.sigmoid(pred[..., 4])
    class_logits = pred[..., 5:]
    class_probs = torch.sigmoid(class_logits)
    confidence = objectness.unsqueeze(-1) * class_probs
    
    print(f"\nObjectness stats:")
    print(f"  Mean: {objectness.mean().item():.4f}")
    print(f"  Max: {objectness.max().item():.4f}")
    print(f"  Min: {objectness.min().item():.4f}")
    print(f"  Cells > 0.5: {(objectness > 0.5).sum().item()}")
    
    print(f"\nClass probability stats:")
    print(f"  Mean: {class_probs.mean().item():.4f}")
    print(f"  Max: {class_probs.max().item():.4f}")
    print(f"  Min: {class_probs.min().item():.4f}")
    
    print(f"\nConfidence stats:")
    print(f"  Mean: {confidence.mean().item():.4f}")
    print(f"  Max: {confidence.max().item():.4f}")
    print(f"  Min: {confidence.min().item():.4f}")
    print(f"  Cells > 0.3: {(confidence > 0.3).sum().item()}")
    print(f"  Cells > 0.5: {(confidence > 0.5).sum().item()}")
    print(f"  Cells > 0.7: {(confidence > 0.7).sum().item()}")


def main():
    parser = argparse.ArgumentParser(description='Test Wildfire Detection Model')
    parser.add_argument('--model', type=str, default='wildfire_detector_best.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset-root', type=str, default='defi1certif-datasets-fire_small',
                        help='Root directory of the dataset')
    parser.add_argument('--image-size', type=int, default=416,
                        help='Input image size (must match training)')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to test from each class')
    parser.add_argument('--inspect', type=str, default=None,
                        help='Path to a single image to inspect raw predictions')
    
    args = parser.parse_args()
    
    if args.inspect:
        print(f"Inspecting raw predictions for: {args.inspect}")
        inspect_raw_predictions(args.model, args.inspect, args.image_size)
    else:
        test_model_on_images(
            args.model,
            args.dataset_root,
            args.image_size,
            args.conf_threshold,
            args.num_samples
        )


if __name__ == "__main__":
    main()

