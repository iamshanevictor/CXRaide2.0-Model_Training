Medical Image Object Detection with SSD300-VGG16
Overview
This repository contains code for training an SSD300-VGG16 model to detect 9 common thoracic abnormalities in chest X-ray images. The implementation includes data preprocessing, model training, and evaluation pipelines.

Dataset Preparation
Source Datasets
NIH Chest X-ray Dataset

VinBigData Chest X-ray Dataset

Preprocessing Steps:
Abnormality Selection
Limited both datasets to 9 common abnormalities:

Cardiomegaly

Pleural thickening

Pulmonary fibrosis

Pleural effusion

Nodule/Mass

Infiltration

(Include all 9 classes used)

Bounding Box Filtering
Selected only images with bounding box annotations

Dataset Combination
Merged NIH and VinBigData datasets

Class Balancing
Applied data sampling techniques to address class imbalance

Image Resizing
Converted all images to 300x300 pixels (SSD300 input requirement)
