# SSD300-VGG16 Model Training with XRAY Dataset

This repository contains code for training an SSD300-VGG16 object detection model on a custom medical imaging dataset combining NIH and VinBig datasets. The implementation includes data preprocessing, model training, and evaluation metrics visualization.

## Overview
- **Dataset**: Combined NIH Chest X-ray and VinBigDatasets (9 common abnormalities)
- **Model**: SSD300 with VGG16 backbone
- **Key Features**:
  - Custom data preprocessing pipeline
  - Class imbalance handling through balanced sampling
  - Multi-label classification with object detection
  - Custom weighted loss function
  - Model evaluation with mAP and ROC metrics

## Dataset Preparation
### Initial Processing (Pre-DataPreprocessing)
1. **Dataset Combination**:
   - Limited both datasets to 9 common abnormalities
   - Selected only images with bounding box annotations
   - Combined NIH and VinBig datasets

2. **Data Balancing**:
   - Performed balanced sampling to address class imbalance
   - Final class distribution:
     - Cardiomegaly
     - Pleural thickening
     - Pulmonary fibrosis
     - Pleural effusion
     - Nodule/Mass
     - Infiltration

3. **Image Processing**:
   - Resized all images to 300x300 pixels (SSD300 requirement)
   - Converted annotations to Pascal VOC format

## Data Preprocessing (`Final_DataPreprocessing.py`)
### Key Steps:
1. **Directory Setup**:
   ```python
   VOCdevkit/
   └── VOC2007/
       ├── Annotations/       # XML annotation files
       ├── ImageSets/Main/    # Train/val splits
       └── JPEGImages/        # Processed images
