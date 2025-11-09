"""
Real-Time Wildfire Detection Application
Uses trained model to detect wildfires from live webcam feed.
"""
import argparse
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model import WildfireYOLO


def preprocess_frame(frame: np.ndarray, image_size: int = 416) -> torch.Tensor:
    """
    Preprocess a frame for model input.
    
    Args:
        frame: OpenCV BGR image [H, W, 3]
        image_size: Target image size
    
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Apply transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform and add batch dimension
    tensor = transform(pil_image).unsqueeze(0)
    
    return tensor


def draw_detections(
    frame: np.ndarray,
    detections: torch.Tensor,
    conf_threshold: float = 0.3
) -> np.ndarray:
    """
    Draw bounding boxes and labels on frame.
    
    Args:
        frame: Original frame [H, W, 3]
        detections: Detections tensor [N, 6] with [x_min, y_min, x_max, y_max, conf, class_id]
        conf_threshold: Confidence threshold for display
    
    Returns:
        Frame with drawn detections
    """
    frame_h, frame_w = frame.shape[:2]
    
    for detection in detections:
        x_min, y_min, x_max, y_max, conf, class_id = detection
        
        # Filter by confidence
        if conf < conf_threshold:
            continue
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x_min * frame_w)
        y1 = int(y_min * frame_h)
        x2 = int(x_max * frame_w)
        y2 = int(y_max * frame_h)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))
        
        # Draw bounding box (red for wildfire)
        color = (0, 0, 255)  # BGR format: red
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label = f"Wildfire: {conf:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA
        )
    
    return frame


def main():
    parser = argparse.ArgumentParser(description='Real-Time Wildfire Detection')
    parser.add_argument('--model', type=str, default='wildfire_detector_best.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--image-size', type=int, default=416,
                        help='Input image size (must match training)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for detections (higher = fewer false positives)')
    parser.add_argument('--smooth-frames', type=int, default=5,
                        help='Number of frames to average for temporal smoothing (reduces jitter)')
    parser.add_argument('--nms-threshold', type=float, default=0.4,
                        help='NMS IoU threshold')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device('cpu')
            print("⚠ CUDA not available, using CPU (inference will be slower)")
    else:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("⚠ Warning: CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    if not torch.cuda.is_available() and device.type == 'cuda':
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    # Create model
    model = WildfireYOLO(
        num_classes=1,
        num_anchors=1,
        image_size=args.image_size
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # If checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and is compatible.")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize webcam
    print(f"Initializing webcam (index {args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nStarting detection...")
    print("Press 'q' to quit")
    print("-" * 50)
    
    frame_count = 0
    # Temporal smoothing: keep track of recent detections
    detection_history = []  # List of recent detections for smoothing
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Preprocess frame
            input_tensor = preprocess_frame(frame, args.image_size).to(device)
            
            # Inference
            with torch.no_grad():
                predictions = model(input_tensor)
                
                # Decode predictions with NMS
                detections = model.decode_predictions(
                    predictions,
                    conf_threshold=args.conf_threshold,
                    nms_threshold=args.nms_threshold
                )
            
            # Draw detections on frame
            has_fire = False
            if len(detections) > 0 and len(detections[0]) > 0:
                # Filter detections by confidence (detections[0] is already a tensor)
                dets_tensor = detections[0]
                confidences = dets_tensor[:, 4]  # Get confidence column
                mask = confidences >= args.conf_threshold
                filtered_dets = dets_tensor[mask]
                
                if len(filtered_dets) > 0:
                    # Apply temporal smoothing to reduce jitter
                    if args.smooth_frames > 1:
                        # Add current detections to history
                        detection_history.append(filtered_dets.cpu().numpy())
                        
                        # Keep only recent frames
                        if len(detection_history) > args.smooth_frames:
                            detection_history.pop(0)
                        
                        # Average boxes across recent frames
                        if len(detection_history) >= 2:
                            # Get the most confident detection from each frame
                            smoothed_dets = []
                            for hist_frame in detection_history:
                                if len(hist_frame) > 0:
                                    # Get the detection with highest confidence
                                    best_idx = np.argmax(hist_frame[:, 4])
                                    smoothed_dets.append(hist_frame[best_idx])
                            
                            if smoothed_dets:
                                # Average the boxes (weighted by confidence)
                                smoothed_dets = np.array(smoothed_dets)
                                weights = smoothed_dets[:, 4]  # Confidence as weights
                                weights = weights / weights.sum()  # Normalize
                                
                                # Weighted average of box coordinates
                                avg_box = np.average(smoothed_dets[:, :4], axis=0, weights=weights)
                                avg_conf = np.average(smoothed_dets[:, 4], weights=weights)
                                
                                # Only use smoothed box if confidence is reasonable
                                if avg_conf >= args.conf_threshold:
                                    smoothed_tensor = torch.tensor([[avg_box[0], avg_box[1], avg_box[2], avg_box[3], avg_conf, 0]], 
                                                                   device=filtered_dets.device, dtype=filtered_dets.dtype)
                                    display_frame = draw_detections(
                                        display_frame,
                                        smoothed_tensor,
                                        conf_threshold=args.conf_threshold
                                    )
                                    has_fire = True
                                    
                                    if frame_count % 10 == 0:  # Print every 10 frames
                                        print(f"Frame {frame_count}: Detected wildfire (smoothed) | "
                                              f"Conf: {avg_conf:.3f}")
                        else:
                            # Not enough history yet, use current detection
                            display_frame = draw_detections(
                                display_frame,
                                filtered_dets,
                                conf_threshold=args.conf_threshold
                            )
                            has_fire = True
                    else:
                        # No smoothing, use current detections directly
                        display_frame = draw_detections(
                            display_frame,
                            filtered_dets,
                            conf_threshold=args.conf_threshold
                        )
                        has_fire = True
                    
                    # Print detection info with confidence scores (less frequently)
                    if not args.smooth_frames > 1 or frame_count % 10 == 0:
                        max_conf = confidences[mask].max().item()
                        avg_conf = confidences[mask].mean().item()
                        print(f"Frame {frame_count}: Detected {len(filtered_dets)} wildfire(s) | "
                              f"Max conf: {max_conf:.3f}, Avg conf: {avg_conf:.3f}")
                else:
                    # Detections were filtered out (below confidence threshold)
                    # Clear history when no detections
                    if len(detection_history) > 0:
                        detection_history.pop(0)
                    
                    if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                        max_conf_below = confidences.max().item() if len(confidences) > 0 else 0.0
                        print(f"Frame {frame_count}: Low confidence detections (max: {max_conf_below:.3f}) - No fire detected")
            else:
                # No detections at all - clear history
                if len(detection_history) > 0:
                    detection_history.pop(0)
            
            # Display status on frame
            status_text = "FIRE DETECTED" if has_fire else "No Fire"
            status_color = (0, 0, 255) if has_fire else (0, 255, 0)  # Red if fire, Green if no fire
            cv2.putText(
                display_frame,
                status_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )
            
            # Add FPS counter (simple version)
            frame_count += 1
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display frame
            cv2.imshow("Wildfire Detection", display_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()

