# Perovskite Solar Cell Defect Detection using Deep Learning

This repository contains the implementation of a comprehensive multi-model deep learning framework for automated defect detection in perovskite solar cell (PSC) scanning electron microscopy (SEM) images. The project benchmarks nine state-of-the-art architectures for classifying five distinct defect types in PSC materials.


## 📋 Project Overview

Perovskite solar cells represent a promising frontier in photovoltaic technology due to their exceptional optoelectronic properties and fabrication simplicity. However, defects in these materials significantly degrade device performance and long-term stability. This project implements and evaluates deep learning models for automated defect classification to enhance quality control in PSC manufacturing and research.

### Defect Categories

The models classify SEM images into five categories:
1. 3D perovskite
2. 3D perovskite with PbI₂ excess
3. 3D perovskite with pinholes
4. 3D-2D mixed perovskite
5. 3D-2D mixed perovskite with pinholes

### Implemented Models

The repository includes implementations of nine architectures:
- YOLOv8 (Ultralytics)
- ResNet50V2
- DenseNet169
- EfficientNetB3
- MobileNetV3 Large
- Vision Transformer (ViT)
- CoCa
- YOLOv9
- InceptionV3

## 🔍 Key Findings

- **YOLOv8** achieved 100% accuracy on the test set with 1,250 images
- **ResNet50V2** and **DenseNet169** followed closely with 96.7% accuracy
- Vision Transformer and other models showed lower performance, highlighting the challenges of limited dataset size for newer architectures
- Data augmentation strategies were essential for mitigating dataset limitations
- The Streamlit web application demonstrates successful deployment for practical use

## 🚀 Getting Started

### Prerequisites

```
Python 3.8+
TensorFlow 2.9+
PyTorch 1.12+
Ultralytics
Streamlit
OpenCV
NumPy
Pandas
scikit-learn
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/username/perovskite-defect-detection.git
cd perovskite-defect-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download and extract the dataset:
```bash
# Extract the provided RAR files in the 'dataset' directory
mkdir -p dataset
# Extract dataset.rar into this directory
```

## 📊 Dataset Information

Due to GitHub storage limitations, the dataset is provided as compressed RAR files. Please extract these files to use the dataset.

**Important Note**: The dataset used in this study is relatively small (2,380-4,560 images) compared to typical deep learning requirements due to the inherent challenges in collecting SEM images of perovskite materials, which require specialized equipment and expertise.

### Dataset Structure

```
dataset/
├── train/
│   ├── pure_3d_perovskite/
│   ├── 3d_perovskite_pbi2_excess/
│   ├── 3d_perovskite_pinholes/
│   ├── 3d_2d_mixed_perovskite/
│   └── 3d_2d_mixed_pinholes/
├── validation/
└── test/
```

### Data Augmentation

The code includes comprehensive data augmentation pipelines to address dataset limitations:
- For standard models: Keras ImageDataGenerator with horizontal/vertical flips, rotations, zoom, brightness shifts, and random shearing
- For YOLOv8: Ultralytics' built-in augmentations (Mosaic, RandomHSV, Flip, etc.)

## 💻 Usage

### Training Models

```bash
# To train multi models
python multi model training 2.py

# To train multi models
python multi model training.py

# To train YOLOv8
pretrain_model.ipynb

```

### Data Augmentation

```bash
python data augmentation.py
```

### Running the Web App

```bash
streamlit run streamlit_app.py
```

## ⚠️ Model Availability

Due to GitHub storage limitations, pre-trained models are not included in this repository. You can:

1. Train the models yourself using the provided code

## 📊 Results

| Model | Test Accuracy (%) | Weighted F1-Score | Training Time |
|-------|------------------|------------------|--------------|
| YOLOv8 (Ultralytics) | 100.0 | 1.000 | 12 min |
| ResNet50V2 | 96.7 | 0.966 | 1h 21m |
| DenseNet169 | 96.7 | 0.966 | 4h 57m |
| YOLOv9 | 45.0 | 0.411 | 7.8 min |
| CoCa | 35.0 | 0.324 | 8.2 min |
| EfficientNetB3 | 33.3 | 0.297 | 47 min |
| Vision Transformer | 31.7 | 0.222 | 13.0 min |
| MobileNetV3 Large | 25.0 | 0.171 | 21 min |
| InceptionV3 | 16.7 | 0.060 | 1h 14m |

## 🖼️ Web Application

The repository includes a Streamlit web application for practical deployment of the defect detection models. The app allows users to:

- Upload SEM images for real-time classification
- View confidence scores for each defect category
- Access reference images for different defect types
- Get explanations for detected defects

<img src="https://github.com/Sahilsonii/images/blob/main/perovskite%20solar%20cell/1.png" alt="PSC Defect Detection"/>
<img src="https://github.com/Sahilsonii/images/blob/main/perovskite%20solar%20cell/2.png" alt="PSC Defect Detection"/>
<img src="https://github.com/Sahilsonii/images/blob/main/perovskite%20solar%20cell/3.png" alt="PSC Defect Detection"/>
<img src="https://github.com/Sahilsonii/images/blob/main/perovskite%20solar%20cell/4.png" alt="PSC Defect Detection"/>

## 🛠️ Future Work

- Collaborative dataset expansion to overcome the inherent limitations in SEM image collection
- Domain adaptation techniques to improve generalization across different microscopy equipment
- Self-supervised learning to leverage unlabeled SEM images
- Active learning strategies to maximize the value of limited labeled data
- Multi-task learning to integrate defect classification with grain size measurement or phase composition analysis

## 🔄 Limitations and Challenges

The primary limitation of this work is the inherent difficulty in collecting large-scale SEM image datasets, which requires:
- Specialized electron microscopy equipment (costing $100,000+)
- Expert microscope operation
- Careful perovskite sample synthesis and preparation
- Time-intensive imaging sessions
- Expert labeling of defect types

These constraints make collecting the 50,000+ images typically recommended for deep learning prohibitively expensive and time-consuming for most research laboratories.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository or contact [sahilsonii369@gmail.com].
