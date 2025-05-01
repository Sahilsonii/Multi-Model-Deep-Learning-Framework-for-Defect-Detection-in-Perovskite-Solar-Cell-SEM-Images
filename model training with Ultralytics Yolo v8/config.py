# -*- coding: utf-8 -*-
"""config.py

Configuration file for perovskite defect classification project, adapted for local execution.
"""

import os
import cv2
import numpy as np

"""Base project directory"""
BASE_DIR = r"C:\Users\ASUS\Desktop\New folder (2)\model training with Ultralytics Yolo v8"
os.makedirs(BASE_DIR, exist_ok=True)

"""Image parameters"""
IMAGE_SIZE = 224
IMG_HEIGHT = IMAGE_SIZE
IMG_WIDTH = IMAGE_SIZE
IMG_CHANNELS = 3

"""Model configurations"""
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
FINE_TUNING_LEARNING_RATE = 1e-5
FINE_TUNING_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10

"""Training parameters"""
TRANSFER_LEARNING = False
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

"""Directory structure"""
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VALIDATION_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

# Output directories
MODEL_DIR = os.path.join(BASE_DIR, "models")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "visualizations")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploads")

# Path to the pre-trained model file
MODEL_PATH = os.path.join(MODEL_DIR, "ResNet50V2_20250410_110048.h5")  # Update with your actual model file

# Create output directories if they don't exist
for dir_path in [MODEL_DIR, VISUALIZATION_DIR, UPLOAD_FOLDER]:
    os.makedirs(dir_path, exist_ok=True)

"""Flask configurations (for web app, not used in training)"""
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
HOST = "0.0.0.0"
PORT = 5000
DEBUG_MODE = True

"""Dynamic class detection"""
def get_defect_classes(train_dir):
    if os.path.exists(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        print("Detected classes:", classes)
        return classes
    else:
        print(f"Warning: {train_dir} not found!")
        return []

DEFECT_CLASSES = get_defect_classes(TRAIN_DIR)

"""Edge masking for image processing"""
def create_edge_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], 0, 255, -1)
        erode_kernel = np.ones((15, 15), np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=1)
    else:
        h, w = mask.shape
        mh, mw = int(h * 0.1), int(w * 0.1)
        mask[mh:h-mh, mw:w-mw] = 255
    return mask

"""Image processing function for prediction and training"""
def process_image_for_prediction(filepath, apply_edge_masking=True):
    try:
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Failed to load image: {filepath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Histogram equalization
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        if apply_edge_masking:
            mask = create_edge_mask(img_resized)
            img_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        img_normalized = img_resized.astype(np.float32) / 255.0
        model_input = np.expand_dims(img_normalized, axis=0)
        return {
            'original_image': img,
            'processed_image': img_resized,
            'model_input': model_input
        }
    except Exception as e:
        print(f"Image processing error: {e}")
        return None