"""
Dataset Download Script
Downloads and extracts the wildfire detection dataset.
"""
import os
import tarfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_small.tar"
DATASET_NAME = "defi1certif-datasets-fire_small.tar"
EXTRACT_DIR = "defi1certif-datasets-fire_small"


def download_dataset(url: str, filename: str):
    """Download the dataset if it doesn't exist."""
    if os.path.exists(filename):
        print(f"Dataset {filename} already exists. Skipping download.")
        return
    
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Download complete: {filename}")


def extract_dataset(tar_path: str, extract_dir: str):
    """Extract the tar archive."""
    if os.path.exists(extract_dir):
        print(f"Directory {extract_dir} already exists. Skipping extraction.")
        return
    
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall()
    print(f"Extraction complete: {extract_dir}")


def inspect_dataset(dataset_dir: str):
    """Inspect the dataset structure."""
    print("\n" + "="*50)
    print("Dataset Structure Inspection")
    print("="*50)
    
    if not os.path.exists(dataset_dir):
        print(f"Error: {dataset_dir} does not exist!")
        return
    
    for root, dirs, files in os.walk(dataset_dir):
        level = root.replace(dataset_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        # Count image files
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            print(f"{subindent}[{len(image_files)} image files]")


if __name__ == "__main__":
    # Download dataset
    download_dataset(DATASET_URL, DATASET_NAME)
    
    # Extract dataset
    extract_dataset(DATASET_NAME, EXTRACT_DIR)
    
    # Inspect structure
    inspect_dataset(EXTRACT_DIR)
    
    print("\nDataset preparation complete!")

