# utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import io
import base64

def enhance_pinholes(image, mask=None):
    """
    Enhance pinhole features for better detection.
    
    Args:
        image: Input image
        mask: Optional mask to restrict processing to specific regions
        
    Returns:
        Binary image with enhanced pinholes, visualization image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Ensure gray is 8-bit unsigned (CV_8UC1)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    
    # Apply mask if provided
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Apply contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Use adaptive thresholding to identify potential pinholes
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Create visualization
    visualization = None
    if len(image.shape) == 3:
        visualization = image.copy()
        # Add red overlay for pinhole regions
        visualization[thresh == 255] = [255, 0, 0]
    
    return thresh, visualization

def generate_cam(model, image, layer_name=None, save_path="static/visualizations/cam_output.jpg"):
    """
    Generate Class Activation Map (CAM) for a given image.

    Args:
        model: Trained Keras model
        image: Preprocessed image tensor (single sample)
        layer_name: Name of the target convolutional layer (optional)
        save_path: Path to save the resulting CAM image

    Returns:
        Path to saved CAM image
    """
    import tensorflow as tf

    # Automatically pick last Conv2D layer if not provided
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError("No Conv2D layer found in the model.")

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(tf.expand_dims(image, axis=0))
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()

    # Convert heatmap to RGB image and save
    import cv2
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # If the input is normalized, undo it for visualization
    original = np.uint8(image * 255) if image.max() <= 1.0 else np.uint8(image)
    if original.shape[-1] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    superimposed_img = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(save_path, superimposed_img)

    return save_path

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Keras history object from model.fit()
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

def plot_sample_predictions(images, true_labels, predictions, class_names, num_samples=5):
    """
    Plot sample predictions.
    
    Args:
        images: Array of images
        true_labels: True labels (one-hot encoded)
        predictions: Model predictions
        class_names: List of class names
        num_samples: Number of samples to plot
        
    Returns:
        Matplotlib figure
    """
    true_indices = np.argmax(true_labels, axis=1)
    pred_indices = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(15, 3 * min(num_samples, len(images))))
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(min(num_samples, len(images)), 3, i*3 + 1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[true_indices[i]]}")
        plt.axis('off')
        
        plt.subplot(min(num_samples, len(images)), 3, i*3 + 2)
        plt.imshow(images[i])
        plt.title(f"Pred: {class_names[pred_indices[i]]}")
        plt.axis('off')
        
        plt.subplot(min(num_samples, len(images)), 3, i*3 + 3)
        plt.bar(range(len(class_names)), predictions[i])
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.title('Probabilities')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

def fig_to_base64(fig_buf):
    """Convert matplotlib figure buffer to base64 for HTML display."""
    return base64.b64encode(fig_buf.getvalue()).decode('utf-8')