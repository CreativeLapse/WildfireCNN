"""
YOLO-based CNN Model for Wildfire Detection
Uses a pre-trained backbone with a YOLO detection head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List


class YOLODetectionHead(nn.Module):
    """
    YOLO detection head that predicts:
    - Bounding box coordinates (x, y, w, h)
    - Objectness score (confidence)
    - Class probability (wildfire)
    """
    
    def __init__(self, in_channels: int, num_anchors: int = 1, num_classes: int = 1):
        """
        Args:
            in_channels: Number of input channels from backbone
            num_anchors: Number of anchor boxes per grid cell
            num_classes: Number of classes (1 for wildfire)
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Each anchor predicts: 4 (bbox) + 1 (objectness) + num_classes (class probs)
        out_channels = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map of shape [B, C, H, W]
        Returns:
            Predictions of shape [B, num_anchors * (5 + num_classes), H, W]
        """
        return self.conv(x)


class WildfireYOLO(nn.Module):
    """
    YOLO-based model for wildfire detection.
    Uses a pre-trained backbone (ResNet-18) with a YOLO detection head.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        num_anchors: int = 1,
        image_size: int = 416,
        backbone: str = "resnet18"
    ):
        """
        Args:
            num_classes: Number of classes (1 for wildfire)
            num_anchors: Number of anchor boxes per grid cell
            image_size: Input image size
            backbone: Backbone architecture ("resnet18" or "mobilenet_v2")
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.image_size = image_size
        
        # Load pre-trained backbone
        if backbone == "resnet18":
            backbone_model = models.resnet18(pretrained=True)
            # Remove the final fully connected layer
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
            backbone_out_channels = 512
        elif backbone == "mobilenet_v2":
            backbone_model = models.mobilenet_v2(pretrained=True)
            self.backbone = backbone_model.features
            backbone_out_channels = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Detection head
        self.detection_head = YOLODetectionHead(
            in_channels=backbone_out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
        # Initialize detection head weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize detection head weights."""
        for m in self.detection_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape [B, 3, H, W]
        Returns:
            Predictions of shape [B, num_anchors * (5 + num_classes), H_out, W_out]
            Where each anchor predicts: [x, y, w, h, obj, class]
        """
        # Extract features
        features = self.backbone(x)
        
        # Detection head
        predictions = self.detection_head(features)
        
        return predictions
    
    def decode_predictions(
        self,
        predictions: torch.Tensor,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[torch.Tensor]:
        """
        Decode raw predictions into bounding boxes.
        
        Args:
            predictions: Raw predictions [B, num_anchors * (5 + num_classes), H, W]
            conf_threshold: Confidence threshold for filtering
            nms_threshold: NMS IoU threshold
        
        Returns:
            List of detections for each image: [x_min, y_min, x_max, y_max, conf, class_id]
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Reshape predictions
        # [B, num_anchors * (5 + num_classes), H, W] -> [B, num_anchors, 5 + num_classes, H, W]
        pred_shape = predictions.shape
        H, W = pred_shape[2], pred_shape[3]
        
        predictions = predictions.view(
            batch_size,
            self.num_anchors,
            5 + self.num_classes,
            H,
            W
        )
        
        # Permute to [B, num_anchors, H, W, 5 + num_classes]
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
        
        # Create grid indices
        grid_y_idx, grid_x_idx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        all_detections = []
        
        for b in range(batch_size):
            detections = []
            
            for a in range(self.num_anchors):
                # Extract predictions for this anchor
                pred = predictions[b, a]  # [H, W, 5 + num_classes]
                
                # Split predictions
                x_offset = torch.sigmoid(pred[..., 0])  # [H, W] offset from grid cell (0-1)
                y_offset = torch.sigmoid(pred[..., 1])  # [H, W] offset from grid cell (0-1)
                width = torch.sigmoid(pred[..., 2])  # [H, W] normalized width (0-1)
                height = torch.sigmoid(pred[..., 3])  # [H, W] normalized height (0-1)
                objectness = torch.sigmoid(pred[..., 4])  # [H, W]
                class_logits = pred[..., 5:]  # [H, W, num_classes]
                class_probs = torch.sigmoid(class_logits)  # [H, W, num_classes]
                
                # Convert to absolute normalized coordinates
                # Match the loss function: x_center = (grid_x_idx + x_offset) / W
                x_center = (grid_x_idx + x_offset) / W
                y_center = (grid_y_idx + y_offset) / H
                
                # Confidence = objectness * class_prob
                confidence = objectness.unsqueeze(-1) * class_probs  # [H, W, num_classes]
                
                # Filter by confidence threshold
                mask = confidence > conf_threshold
                
                if mask.any():
                    # Get indices where confidence > threshold
                    indices = torch.nonzero(mask, as_tuple=False)
                    
                    for idx in indices:
                        h_idx, w_idx, c_idx = idx[0].item(), idx[1].item(), idx[2].item()
                        
                        conf = confidence[h_idx, w_idx, c_idx].item()
                        x_c = x_center[h_idx, w_idx].item()
                        y_c = y_center[h_idx, w_idx].item()
                        w = width[h_idx, w_idx].item()
                        h = height[h_idx, w_idx].item()
                        
                        # Convert center, width, height to x_min, y_min, x_max, y_max
                        x_min = max(0, x_c - w / 2)
                        y_min = max(0, y_c - h / 2)
                        x_max = min(1, x_c + w / 2)
                        y_max = min(1, y_c + h / 2)
                        
                        detections.append([x_min, y_min, x_max, y_max, conf, c_idx])
            
            if detections:
                detections = torch.tensor(detections, device=device)
                # Apply NMS
                detections = self._apply_nms(detections, nms_threshold)
            else:
                detections = torch.zeros((0, 6), device=device)
            
            # If we have detections, try to create tighter bounding boxes based on confidence heatmap
            # This helps when the model was trained with full-image boxes but we want to localize the fire
            if len(detections) > 0:
                # Get confidence heatmap for this image
                pred = predictions[b, 0]  # [H, W, 5 + num_classes]
                objectness = torch.sigmoid(pred[..., 4])  # [H, W]
                class_logits = pred[..., 5:]  # [H, W, num_classes]
                class_probs = torch.sigmoid(class_logits)  # [H, W, num_classes]
                confidence = objectness.unsqueeze(-1) * class_probs  # [H, W, num_classes]
                confidence_2d = confidence.squeeze(-1)  # [H, W]
                
                # Check if any original detection was full-screen (width/height > 0.9)
                original_is_fullscreen = False
                for det in detections:
                    box_w = det[2].item() - det[0].item()
                    box_h = det[3].item() - det[1].item()
                    if box_w > 0.9 or box_h > 0.9:
                        original_is_fullscreen = True
                        break
                
                # If original was full-screen, try to find the actual fire location
                if original_is_fullscreen and confidence_2d.max().item() > conf_threshold:
                    # Find the single highest confidence point (peak of the heatmap)
                    # This is more stable than weighted center when confidence is uniform
                    max_val, max_idx = torch.max(confidence_2d.flatten(), dim=0)
                    max_y_idx = max_idx.item() // W
                    max_x_idx = max_idx.item() % W
                    
                    # Also calculate weighted center for comparison
                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(H, device=device, dtype=torch.float32),
                        torch.arange(W, device=device, dtype=torch.float32),
                        indexing='ij'
                    )
                    x_coords_norm = x_coords / W
                    y_coords_norm = y_coords / H
                    
                    # Use only high-confidence cells for weighted center (top 20% of confidence)
                    conf_sorted, _ = torch.sort(confidence_2d.flatten(), descending=True)
                    top_k = max(1, int(confidence_2d.numel() * 0.2))
                    top_k_threshold = conf_sorted[top_k - 1].item()
                    high_conf_mask = confidence_2d > top_k_threshold
                    
                    if high_conf_mask.any():
                        # Calculate weighted center using only high-confidence cells
                        high_conf_values = confidence_2d[high_conf_mask]
                        total_conf = high_conf_values.sum()
                        
                        if total_conf > 0:
                            center_x = (confidence_2d[high_conf_mask] * x_coords_norm[high_conf_mask]).sum() / total_conf
                            center_y = (confidence_2d[high_conf_mask] * y_coords_norm[high_conf_mask]).sum() / total_conf
                            
                            # Use the peak location if it's significantly different from center
                            # Otherwise use weighted center
                            peak_x = (max_x_idx + 0.5) / W
                            peak_y = (max_y_idx + 0.5) / H
                            
                            # Check if peak is far from weighted center
                            dist_from_center = ((peak_x - center_x)**2 + (peak_y - center_y)**2)**0.5
                            
                            # If peak is far (>0.1), use peak; otherwise use weighted center
                            if dist_from_center > 0.1:
                                final_center_x = peak_x
                                final_center_y = peak_y
                            else:
                                final_center_x = center_x
                                final_center_y = center_y
                            
                            # Calculate box size based on spread of high-confidence regions
                            y_indices, x_indices = torch.where(high_conf_mask)
                            if len(y_indices) > 0:
                                spread_x = (x_indices.max().item() - x_indices.min().item() + 1) / W
                                spread_y = (y_indices.max().item() - y_indices.min().item() + 1) / H
                                
                                # Use adaptive box size (30-50% of image, based on spread)
                                box_w = max(0.3, min(0.5, spread_x * 1.2))
                                box_h = max(0.3, min(0.5, spread_y * 1.2))
                            else:
                                # Default box size if no spread
                                box_w = 0.4
                                box_h = 0.4
                            
                            # Create box centered at final center
                            x_min_norm = max(0, final_center_x - box_w / 2)
                            y_min_norm = max(0, final_center_y - box_h / 2)
                            x_max_norm = min(1, final_center_x + box_w / 2)
                            y_max_norm = min(1, final_center_y + box_h / 2)
                            
                            # Ensure minimum box size
                            if x_max_norm - x_min_norm < 0.2:
                                x_min_norm = max(0, final_center_x - 0.1)
                                x_max_norm = min(1, final_center_x + 0.1)
                            if y_max_norm - y_min_norm < 0.2:
                                y_min_norm = max(0, final_center_y - 0.1)
                                y_max_norm = min(1, final_center_y + 0.1)
                            
                            # Get max confidence
                            max_conf = confidence_2d.max().item()
                            
                            # Replace detections with tighter bounding box
                            detections = torch.tensor([[x_min_norm, y_min_norm, x_max_norm, y_max_norm, max_conf, 0]], device=device)
            
            all_detections.append(detections)
        
        return all_detections
    
    def _apply_nms(self, detections: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """
        Apply Non-Maximum Suppression.
        
        Args:
            detections: [N, 6] tensor with [x_min, y_min, x_max, y_max, conf, class_id]
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        sorted_indices = torch.argsort(detections[:, 4], descending=True)
        detections = detections[sorted_indices]
        
        keep = []
        while len(detections) > 0:
            # Keep the first (highest confidence) detection
            keep.append(detections[0])
            
            if len(detections) == 1:
                break
            
            # Calculate IoU with remaining detections
            ious = self._calculate_iou(detections[0:1], detections[1:])
            
            # Remove detections with IoU > threshold
            mask = ious < iou_threshold
            detections = detections[1:][mask]
        
        if keep:
            return torch.stack(keep)
        else:
            return torch.zeros((0, 6), device=detections.device)
    
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between boxes.
        
        Args:
            box1: [1, 6] or [N, 6] tensor
            box2: [M, 6] tensor
        
        Returns:
            IoU values [M] or [N, M]
        """
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        x2_min, y2_min, x2_max, y2_max = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        
        # Calculate intersection
        inter_x_min = torch.max(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
        inter_y_min = torch.max(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
        inter_x_max = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
        inter_y_max = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))
        
        inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * torch.clamp(inter_y_max - inter_y_min, min=0)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        
        # Return IoU for each box2 with box1
        if iou.shape[0] == 1:
            return iou.squeeze(0)
        return iou


if __name__ == "__main__":
    # Test the model
    model = WildfireYOLO(num_classes=1, num_anchors=1, image_size=416)
    
    # Test forward pass
    x = torch.randn(2, 3, 416, 416)
    with torch.no_grad():
        predictions = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {predictions.shape}")
        
        # Test decoding
        detections = model.decode_predictions(predictions, conf_threshold=0.3)
        print(f"\nDetections for batch:")
        for i, dets in enumerate(detections):
            print(f"  Image {i}: {len(dets)} detections")
            if len(dets) > 0:
                print(f"    First detection: {dets[0]}")

