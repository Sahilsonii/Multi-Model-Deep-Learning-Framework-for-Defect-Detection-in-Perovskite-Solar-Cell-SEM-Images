# Multi-Model Deep Learning Framework for Perovskite Defect Detection

[![Nature Scientific Reports](https://img.shields.io/badge/Published%20in-Nature%20Scientific%20Reports-blue)](https://www.nature.com/articles/s41598-025-25848-x)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--025--25848--x-green)](https://doi.org/10.1038/s41598-025-25848-x)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

> **Automated SEM-based defect detection in FAPbI₃ Perovskite thin films using ResNet50V2, DenseNet169, and YOLOv9**

---

## 📄 Publication

**Title:** A multi-model deep learning framework for SEM-based defect detection in FAPbI₃ Perovskite thin films

**Authors:** Ansari, Z.A., Soni, S., Fatima, S., et al.

**Journal:** Scientific Reports (Nature Portfolio), Volume 15, Article 41909 (2025)

**Citation:**
```bibtex
@article{ansari2025multimodel,
  title={A multi-model deep learning framework for SEM-based defect detection in FAPbI3 Perovskite thin films},
  author={Ansari, Z.A. and Soni, S. and Fatima, S. and others},
  journal={Scientific Reports},
  volume={15},
  pages={41909},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-025-25848-x}
}
```

---

## 🎯 Overview

Perovskite solar cells (PSCs) based on formamidinium lead iodide (FAPbI₃) have demonstrated power conversion efficiencies exceeding 25%, positioning them as leading candidates for next-generation photovoltaics. However, structural defects such as pinholes, PbI₂ accumulation, and grain boundary irregularities significantly compromise device efficiency and stability.

This project presents an **automated, high-precision deep learning framework** for defect classification in mixed-dimensionality FAPbI₃ perovskite films using scanning electron microscopy (SEM) images.

### Key Achievements
- **96.7% Test Accuracy** with ResNet50V2 and DenseNet169
- **Real-time Detection** with YOLOv9 (8-minute training time)
- **Deployed Web Application** for practical laboratory use
- **Published in Nature Scientific Reports** (2025)

---

## 🔬 Research Problem

Conventional defect characterization using SEM is:
- **Labor-intensive** and time-consuming
- **Subjective** and prone to human error
- **Unsuitable** for large-scale quality control

Our solution provides automated, objective, and scalable defect detection suitable for both research and industrial applications.

---

## 📊 Defect Categories

The framework classifies five critical defect types in perovskite thin films:

1. **Pure 3D Perovskite** (reference baseline)
2. **3D Perovskite with PbI₂ Excess**
3. **3D Perovskite with Pinholes**
4. **3D-2D Mixed Perovskite**
5. **3D-2D Mixed Perovskite with Pinholes**

---

## 🏗️ Architecture

### Multi-Model Approach

We benchmarked three complementary deep learning architectures:

#### 1. **ResNet50V2**
- Residual network with skip connections
- Mitigates vanishing gradients
- Excellent for fine-grained texture analysis
- **Performance:** 96.7% accuracy, F1-score: 0.966
- **Training Time:** 81 minutes

#### 2. **DenseNet169**
- Densely connected convolutional network
- Maximizes feature reuse and parameter efficiency
- Robust under limited data conditions
- **Performance:** 96.7% accuracy, F1-score: 0.966
- **Training Time:** 297 minutes

#### 3. **YOLOv9**
- State-of-the-art real-time object detection
- GELAN backbone with PGI module
- Optimized for speed and computational efficiency
- **Performance:** 45.0% accuracy, F1-score: 0.411
- **Training Time:** 8 minutes ⚡

---

## 📈 Results Summary

| Model | Test Accuracy | Weighted F1-Score | Training Time | Use Case |
|-------|--------------|-------------------|---------------|----------|
| **ResNet50V2** | 96.7% | 0.966 | 81 min | High-precision research |
| **DenseNet169** | 96.7% | 0.966 | 297 min | Robust classification |
| **YOLOv9** | 45.0% | 0.411 | 8 min | Real-time industrial QC |

### Key Findings
- **Classification networks** (ResNet50V2, DenseNet169) excel at fine-grained defect identification
- **Detection framework** (YOLOv9) prioritizes computational efficiency for rapid deployment
- Transfer learning and data augmentation effectively overcome limited dataset size (2,380 images)

---

## 🛠️ Technical Implementation

### Dataset
- **Total Images:** 2,380 SEM images (224×224 pixels)
- **Distribution:** 452 training, 12 validation, 12 testing per class
- **Preprocessing:** Normalization, resizing, extensive data augmentation

### Training Configuration
- **Loss Function:** Categorical cross-entropy
- **Optimizer:** Adam (learning rate: 1e-4)
- **Epochs:** 50 with early stopping
- **Transfer Learning:** ImageNet pre-trained weights (ResNet50V2, DenseNet169)

### Hardware
- **CPU:** Intel i7-11800H
- **GPU:** NVIDIA RTX 3050 Ti

---

## 🚀 Features

### 1. Automated Defect Detection
- Eliminates subjective manual analysis
- Provides objective, reproducible results
- Suitable for large-scale quality control

### 2. Multi-Model Framework
- Complementary architectures for different use cases
- Balance between accuracy and computational efficiency
- Flexible deployment options

### 3. Streamlit Web Application
- User-friendly interface for researchers and technicians
- Real-time defect prediction with confidence scores
- Visual overlays and reference image gallery
- No machine learning expertise required

### 4. Practical Deployment
- Bridges gap between research and industrial application
- Enables rapid decision-making and process optimization
- Supports scalable manufacturing quality control

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/perovskite-defect-detection.git
cd perovskite-defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
streamlit>=1.20.0
pillow>=9.0.0
pandas>=1.4.0
```

---

## 💻 Usage

### Training Models

```python
# Train ResNet50V2
python train_resnet.py --epochs 50 --batch_size 32 --learning_rate 1e-4

# Train DenseNet169
python train_densenet.py --epochs 50 --batch_size 32 --learning_rate 1e-4

# Train YOLOv9
python train_yolo.py --epochs 50 --batch_size 16
```

### Running Predictions

```python
from models import load_trained_model
from preprocessing import preprocess_image

# Load model
model = load_trained_model('resnet50v2')

# Predict on new SEM image
image = preprocess_image('path/to/sem_image.png')
prediction = model.predict(image)
defect_class = get_defect_label(prediction)
confidence = get_confidence_score(prediction)

print(f"Detected: {defect_class} (Confidence: {confidence:.2%})")
```

### Streamlit Web Application

```bash
# Launch the web application
streamlit run app.py
```

Access the application at `http://localhost:8501`

---

## 📊 Model Performance

### Confusion Matrix Analysis

**DenseNet169** demonstrates near-perfect classification with minimal confusion between morphologically similar defects (3D perovskite vs. 3D perovskite with PbI₂ excess).

**ResNet50V2** exhibits comparable performance with strong diagonal dominance in the confusion matrix.

**YOLOv9** excels at detecting 3D-2D mixed perovskite with pinholes (F1=0.645) but struggles with subtle texture variations.

### Per-Class Performance

| Defect Type | ResNet50V2 F1 | DenseNet169 F1 | YOLOv9 F1 |
|-------------|---------------|----------------|-----------|
| Pure 3D Perovskite | 0.98 | 0.99 | 0.52 |
| 3D + PbI₂ Excess | 0.95 | 0.96 | 0.15 |
| 3D + Pinholes | 0.97 | 0.98 | 0.41 |
| 3D-2D Mixed | 0.98 | 0.99 | 0.38 |
| 3D-2D + Pinholes | 0.96 | 0.97 | 0.65 |

---

## 🎓 Scientific Contribution

### Novel Aspects
1. **First comprehensive multi-model framework** for FAPbI₃ defect detection
2. **Deployment-ready solution** with practical web application
3. **Addresses dataset scarcity** through transfer learning and augmentation
4. **Benchmarking three architectures** for accuracy-efficiency trade-offs

### Impact
- Accelerates perovskite solar cell optimization
- Enables scalable quality control in manufacturing
- Reduces manual labor and subjective analysis
- Supports commercialization of PSC technologies

---

## 📂 Project Structure

```
perovskite-defect-detection/
├── data/
│   ├── raw/                    # Original SEM images
│   ├── processed/              # Preprocessed images
│   └── augmented/              # Augmented dataset
├── models/
│   ├── resnet50v2/            # ResNet50V2 architecture
│   ├── densenet169/           # DenseNet169 architecture
│   ├── yolov9/                # YOLOv9 detection model
│   └── trained/               # Saved model weights
├── src/
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── augmentation.py        # Data augmentation functions
│   ├── train.py              # Training scripts
│   ├── evaluate.py           # Model evaluation
│   └── predict.py            # Inference utilities
├── webapp/
│   ├── app.py                # Streamlit application
│   ├── utils.py              # Helper functions
│   └── assets/               # UI resources
├── requirements.txt
├── README.md
└── LICENSE
```

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

---

## 🔮 Future Work

### Improvements
1. **Knowledge Distillation** - Transfer representational power to lighter models
2. **Model Pruning & Quantization** - Reduce parameters while retaining accuracy
3. **Semi-supervised Learning** - Leverage large volumes of unlabeled SEM data
4. **High-Resolution Imaging** - Detect nanoscale defects (<100 nm)

### Extensions
1. **Additional Defect Categories** - Voids, cracks, interfacial delamination
2. **Multi-scale Analysis** - Combine different magnification levels
3. **Real-time Monitoring** - Integration with manufacturing lines
4. **Cross-material Generalization** - Extend to other perovskite compositions

---

## 📖 Limitations

- **Dataset Size:** Limited to 2,380 images due to labor-intensive SEM acquisition
- **Pinhole Detection:** Minimum detectable size ~500 nm (well-fabricated films have <100 nm pinholes)
- **Morphological Similarity:** Subtle confusion between 3D perovskite and 3D + PbI₂ excess
- **Generalization:** Trained on specific fabrication conditions

---

## 🤝 Contributing

We welcome contributions from the research community!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions, collaborations, or dataset access requests:

---

## 🙏 Acknowledgments

This work is supported by the **Research Support Fund (RSF)** of Symbiosis International (Deemed University), Pune, India.

We thank the research community for valuable discussions and feedback.

---

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.

- ✅ Non-commercial use
- ✅ Sharing and distribution
- ❌ Commercial use without permission
- ❌ Derivative works

See [LICENSE](LICENSE) for details.

---

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Made with ❤️ for the perovskite solar cell research community**
