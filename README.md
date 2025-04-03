Thoracic Abnormality Detection with SSD300-VGG16 🩺🔍
A deep learning system for detecting 9 common chest abnormalities in X-ray images using PyTorch and SSD architecture.

Overview
-- Combines NIH and VinBigData datasets for robust training
-- Implements SSD300-VGG16 model for object detection
-- Includes full pipeline from data preprocessing to metrics visualization

Features
-- Multi-Dataset Integration

Combines 9 common abnormalities from NIH and VinBigData

Filters images with bounding box annotations
-- Advanced Preprocessing

Converts images to 300x300 resolution

Generates PASCAL VOC format annotations

Balances class distribution through smart sampling
-- Custom Training

Class-weighted loss function for imbalance handling

NMS post-processing with adjustable IoU threshold

LR scheduling via ReduceLROnPlateau
