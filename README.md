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
  - Weighted Box Fusion for Images that has multiple annotations

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
2. **File Operations**:
   - Image Handling:
     - Copy Operations:
       ```python
       # Copies PNG images from training/validation folders to JPEGImages
        shutil.copy(os.path.join(train_folder, filename), jpeg_images_folder)
      - Handles both training (balance_TRAINING) and validation (balance_VALIDATION) sets
      - Maintains original filenames during copy operations
   - Dataset Splits:
     - File List Generation:
       ```python
       # Creates trainval.txt and test.txt for dataset splits
        with open(os.path.join(image_sets_folder, 'trainval.txt'), 'w') as f:
            for filename in trainval_filenames:
                f.write(f"{filename}\n")
      - Generates standardized VOC format split files
      - Uses unique image IDs from CSV files
3. **Annotation Generation**:
   - Processes CSV files containing bounding box information
   - Creates Pascal VOC-style XML annotations using ElementTree
  
## Model Training (Model_Training.py)
### Architecture & Training Details
  - Base Model: SSD300-VGG16 (pretrained on COCO)
  - Modifications:
    - Custom classification head for 7 classes (6 abnormalities + background)
    - Class weights for imbalanced data handling
  - Training Parameters:
    - Batch size: 64 (training), 1 (validation)
    - Optimizer: SGD (lr=0.0001, momentum=0.9)
    - Scheduler: ReduceLROnPlateau
    - Epochs: 300
### Key Features:
  - Custom Dataset Class:
    - Handles both bounding boxes and multi-label classification
    - Includes error handling for corrupted samples
  - Custom Loss Function:
    ```python
    def custom_loss_fn(loss_dict, targets, class_weights):
    # Combines weighted classification loss and regression loss
  - Evaluation Metrics:
    - Mean Average Precision (mAP) at different IoU thresholds
    - Average Recall (AR) for various object sizes
## Metrics Visualization (Metrics_Visualization.py)
### Included Visualizations:
1. **ROC Curves**:
   - Class-specific ROC curves with AUC scores
   - Aggregate performance visualization
2. **Training Loss Plot**:
   - Total loss progression over 260 epochs
   - Highlighted model checkpoints every 20 epochs
3. **Performance Metrics**:
   - mAP@0.50:0.95 and mAP@0.50
   - Average Recall (AR) across different object sizes

### Directory Structure:
- 
   ```python
   /content/drive/Shareddrives/cxraide/ssd300_vgg16/
    ├── VER6/                          # Preprocessed data
    │   ├── balance_TRAINING/
    │   ├── balance_VALIDATION/
    │   └── VOCdevkit/
    ├── VERSION_3/                     # Training artifacts
    │   ├── EXPORT/model_v1/          # Saved models
    │   ├── new_balanced_train(V5).csv
    │   └── balanced_TRAINtrans.csv    # Multi-labels
    └── VERSION_2/                     # Validation data     
 
### Requirements:
-
   ```python
   pip install torch torchvision pillow torchmetrics

### Notes
1. **Class Indices**:
   ```python
   class_to_idx = {
    "Cardiomegaly": 1,
    "Pleural thickening": 2,
    "Pulmonary fibrosis": 3,
    "Pleural effusion": 4,
    "Nodule/Mass": 5,
    "Infiltration": 6
    }
2. **Data Balancing**:
   - Achieved through stratified sampling based on abnormality prevalence
   - Class weights calculated as inverse frequency:
     ```python
     class_weights = {cls: total_samples/count for cls, count in class_counts.items()}
3. **Model Checkpoints**:
   - Saved every 20 epochs
   - Includes loss values and evaluation metrics at each checkpoint
