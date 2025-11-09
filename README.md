# WildfireCNN

A real-time wildfire detection system using YOLO-based CNN architecture with PyTorch. This project detects wildfires from live webcam feeds and draws bounding boxes around detected fire regions.

## Features

- **Real-time Detection**: Uses live webcam feed for wildfire detection
- **YOLO-based Architecture**: Custom YOLO implementation with ResNet-18 backbone
- **Bounding Box Localization**: Accurately localizes fire regions in images
- **Temporal Smoothing**: Reduces jitter in bounding boxes across frames
- **GPU Support**: Automatically uses CUDA if available

## Project Structure

```
WildfireCNN/
├── model.py              # YOLO model architecture
├── dataset.py            # PyTorch dataset loader (supports YOLO format)
├── train.py              # Training script with YOLO loss function
├── detect.py             # Real-time webcam detection application
├── test_model.py         # Model evaluation script
├── download.py           # Dataset download utility
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/WildfireCNN.git
cd WildfireCNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on your dataset:

```bash
python train.py --dataset-root data_bb --epochs 50 --batch-size 16
```

**Arguments:**
- `--dataset-root`: Path to dataset directory (YOLO format: train/images, train/labels, etc.)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate (default: 0.001)
- `--image-size`: Input image size (default: 416)

### Real-Time Detection

Run the webcam detection application:

```bash
python detect.py --model wildfire_detector_best.pth --conf-threshold 0.5 --smooth-frames 5
```

**Arguments:**
- `--model`: Path to trained model checkpoint
- `--conf-threshold`: Confidence threshold (default: 0.5)
- `--smooth-frames`: Number of frames for temporal smoothing (default: 5)
- `--camera`: Camera index (default: 0)

### Testing

Test the model on validation set:

```bash
python test_model.py --model wildfire_detector_best.pth --dataset-root data_bb --num-samples 20
```

## Dataset Format

The project supports two dataset formats:

1. **YOLO Format** (recommended):
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```
   Label files contain: `class_id x_center y_center width height` (normalized)

2. **Old Format** (for backward compatibility):
   ```
   dataset/
   ├── fire/
   ├── no_fire/
   └── start_fire/
   ```

## Model Architecture

- **Backbone**: ResNet-18 (pre-trained on ImageNet)
- **Detection Head**: Custom YOLO head with:
  - Bounding box regression (x, y, w, h)
  - Objectness prediction
  - Class probability (wildfire)
- **Loss Function**: Composite YOLO loss:
  - Coordinate loss (MSE)
  - Objectness loss (BCE)
  - No-object loss (BCE)
  - Classification loss (BCE)

## Requirements

- Python 3.9+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- opencv-python 4.5.0+
- numpy 1.21.0+
- Pillow 8.3.0+
- tqdm 4.62.0+

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: Roboflow Wildfire Detection Dataset
- Model architecture inspired by YOLO (You Only Look Once)

