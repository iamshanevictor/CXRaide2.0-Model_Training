# X-RAY Abnormality Detection with SSD300-VGG16 🔍

A deep learning system to detect 9 chest abnormalities in X-ray images using PyTorch and SSD architecture.

## Overview  
- Combines NIH and VinBigData datasets for robust training  
- Implements SSD300-VGG16 model for object detection  
- Full pipeline: data preprocessing → training → metrics visualization  

## Features  
- **Transfer Learning**: Pre-trained SSD300-VGG16 backbone  
- **Data Preprocessing**: 300x300 resolution conversion + VOC format annotations  
- **Class Balancing**: Smart sampling for imbalanced abnormalities  
- **Metrics**: mAP@[0.5:0.95], ROC-AUC, AR@100  
- **Visualization**: ROC curves, training loss graphs  

## Requirements  
- Python 3.8+  
- torch==2.0.1  
- torchvision==0.15.2  
- pandas  
- matplotlib  
- scikit-learn  
- Pillow  

## Model Architecture  
- Backbone: Modified VGG16  
- Feature Pyramid: 6 multi-scale feature maps  
- Default Boxes: 8732 per image  
- Heads: 7-class classifier + 4-coordinate regressor  

## Training Configuration  
- **Optimizer**: SGD (lr=0.0001, momentum=0.9)  
- **Batch Sizes**: 64 (train), 1 (val)  
- **Epochs**: 300  
- **Checkpoints**: Saved every 20 epochs  

## Dataset Structure  
- **Annotations**: XML files in PASCAL VOC format  
- **Splits**:  
  - trainval.txt (70% images)  
  - test.txt (30% images)  
- **Directory**:  
