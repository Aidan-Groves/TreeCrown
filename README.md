# Tree Crown Detection with Mask R-CNN
This project uses a deep learning-based computer vision pipeline to detect and delineate individual tree crowns from aerial imagery. Leveraging the Mask R-CNN architecture, the model is trained to segment trees at the crown level, aiding ecological monitoring, forest management, and remote sensing analysis.

# Project Overview
The notebook walks through the implementation of a tree crown detection system using Mask R-CNN. Aerial or satellite imagery is processed and passed through a pre-trained neural network, which identifies tree crowns as distinct instances, returning both bounding boxes and precise masks for segmentation.

This project is especially relevant for ecological applications, where accurately quantifying tree counts, canopy area, and spatial distribution supports conservation and land-use decisions.

# Features:
Fine-tuning of Mask R-CNN on aerial imagery with labeled tree crowns
Data loading and augmentation for geospatial image formats
Visualization of predicted masks, confidence scores, and bounding boxes
Use of pre-trained weights for transfer learning
Evaluation metrics for segmentation accuracy

# Dependencies:
Python 3.10+
PyTorch
torchvision
OpenCV
numpy
matplotlib
pycocotools (for COCO-style mask handling)

# Limitations:
Performance may vary depending on the resolution and clarity of input imagery
Requires GPU acceleration for reasonable training and inference times
Ground truth data in COCO format is assumed for training

Future Improvements:
Integrate NDVI or multispectral data as additional channels
Deploy as a web-based interface for visualizing predictions on new imagery
Explore lightweight segmentation models for edge deployment (e.g., on drones)
