# Perovskite Solar Cell Defect Detection using Deep Learning

This repository contains the implementation of a comprehensive multi-model deep learning framework for automated defect detection in perovskite solar cell (PSC) scanning electron microscopy (SEM) images. The project benchmarks nine state-of-the-art architectures for classifying five distinct defect types in PSC materials.

![PSC Defect Detection](https://raw.githubusercontent.com/username/repository/main/images/sample_defects.png)

## 📋 Project Overview

Perovskite solar cells represent a promising frontier in photovoltaic technology due to their exceptional optoelectronic properties and fabrication simplicity. However, defects in these materials significantly degrade device performance and long-term stability. This project implements and evaluates deep learning models for automated defect classification to enhance quality control in PSC manufacturing and research.

### Defect Categories

The models classify SEM images into five categories:
1. Pure 3D perovskite
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
# To train ResNet50V2
python train_resnet.py

# To train DenseNet169
python train_densenet.py

# To train YOLOv8
python train_yolov8.py

# For other models
python train_model.py --model [model_name]
```

### Data Augmentation

```bash
python augment_data.py --input_dir dataset/train --output_dir dataset/augmented
```

### Evaluation

```bash
python evaluate.py --model [path_to_model] --test_dir dataset/test
```

### Running the Web App

```bash
streamlit run app.py
```

## 📁 Repository Structure

```
.
├── README.md
├── requirements.txt
├── models/
│   ├── train_resnet.py
│   ├── train_densenet.py
│   ├── train_yolov8.py
│   └── train_model.py
├── utils/
│   ├── data_loader.py
│   ├── augmentation.py
│   └── evaluation.py
├── app/
│   ├── app.py
│   └── components/
└── dataset/
    ├── dataset.rar  # Extract this to use the dataset
    └── README_dataset.md
```

## ⚠️ Model Availability

Due to GitHub storage limitations, pre-trained models are not included in this repository. You can:

1. Train the models yourself using the provided code
2. Request the models by creating an issue in this repository
3. Access a subset of the models via [Google Drive](https://drive.google.com/folder/link) (if available)

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

![Web App Screenshot]([https://raw.githubusercontent.com/username/repository/main/images/webapp_screenshot.png](https://github.com/Sahilsonii/images/blob/main/perovskite%20solar%20cell/1.png))
![Web App Screenshot](https://raw.githubusercontent.com/username/repository/main/images/webapp_screenshot.png)
![Web App Screenshot](https://raw.githubusercontent.com/username/repository/main/images/webapp_screenshot.png)
![Web App Screenshot](https://raw.githubusercontent.com/username/repository/main/images/webapp_screenshot.png)

## 📚 Citation

If you use this code or dataset in your research, please cite:

```
@article{author2025comprehensive,
  title={Comprehensive Multi-Model Deep Learning Framework for Automated Defect Detection in Perovskite Solar Cell SEM Images},
  author={Author, A.},
  journal={Journal Name},
  year={2025},
  volume={},
  pages={}
}
```

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
