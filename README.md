# Geometry-Aware Road-Marking Segmentation

Geometry-Aware Modeling and Structure-Adaptive Multi-Scale Feature Fusion for Semantic Segmentation of Road-Marking Point Clouds.

## Code Description

- `train.py`: Model training, weighted focal loss, learning rate scheduling, and performance evaluation.
- `config.py`: Network configuration and training parameters.
- `las_to_txt.py`: Conversion of LAS point clouds to TXT format with XYZ coordinates, RGB colors, intensity, and semantic labels.
- `npy_to_las.py`: Conversion of NPY point clouds to LAS format for visualization.

## Environment

- Python 3.9.19
- TensorFlow 2.10.0
- NumPy
- laspy
