# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import config
# from scipy.stats import entropy

# # Thresholds for confidence and out-of-distribution detection
# CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for accepting predictions
# ENTROPY_THRESHOLD = 1.2     # Entropy threshold for out-of-distribution detection

# def load_trained_model():
#     """
#     Load the trained model from saved path.
    
#     Returns:
#         Keras model
#     """
#     try:
#         model = load_model(config.MODEL_PATH)
#         print(f"Model loaded successfully from {config.MODEL_PATH}")
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None
    
# def calculate_prediction_entropy(probabilities):
#     """
#     Calculate entropy of prediction probabilities.
#     Higher entropy indicates uncertainty (possibly OOD).
    
#     Args:
#         probabilities: Array of class probabilities
        
#     Returns:
#         Entropy value
#     """
#     # Add small epsilon to avoid log(0)
#     epsilon = 1e-10
#     probabilities = np.clip(probabilities, epsilon, 1.0)
#     return -np.sum(probabilities * np.log(probabilities))
    
# def predict_defect(model, img_processed):
#     """
#     Predicts the defect class for a given input image.
#     Applies confidence-based filtering and out-of-distribution detection.
    
#     Args:
#         model: Loaded Keras model
#         img_processed: Preprocessed image tensor (batch of 1)
        
#     Returns:
#         Dictionary with prediction results and confidence metrics
#     """
#     if model is None:
#         return {
#             'defect_type': 'Unknown - Model not loaded',
#             'confidence': 0.0,
#             'is_out_of_distribution': True,
#             'entropy': 0.0
#         }
    
#     try:
#         # Get prediction probabilities
#         pred_probs = model.predict(img_processed, verbose=0)
        
#         # Calculate metrics
#         predicted_class_index = np.argmax(pred_probs[0])
#         confidence = float(pred_probs[0][predicted_class_index])
#         pred_entropy = calculate_prediction_entropy(pred_probs[0])
        
#         print(f"Raw predictions: {pred_probs[0]}")
#         print(f"Predicted class index: {predicted_class_index}, Confidence: {confidence:.4f}, Entropy: {pred_entropy:.4f}")
        
#         # Determine if prediction is likely out-of-distribution
#         is_out_of_distribution = (confidence < CONFIDENCE_THRESHOLD) or (pred_entropy > ENTROPY_THRESHOLD)
        
#         if is_out_of_distribution:
#             if confidence < CONFIDENCE_THRESHOLD:
#                 defect_type = "Uncertain - Low confidence"
#             else:
#                 defect_type = "Uncertain - Possibly out-of-distribution"
#         else:
#             defect_type = config.DEFECT_CLASSES[predicted_class_index]
        
#         return {
#             'defect_type': defect_type,
#             'confidence': confidence,
#             'is_out_of_distribution': is_out_of_distribution,
#             'entropy': pred_entropy,
#             'raw_predictions': pred_probs[0].tolist(),
#             'class_index': int(predicted_class_index)
#         }
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return {
#             'defect_type': f'Error during prediction: {str(e)}',
#             'confidence': 0.0,
#             'is_out_of_distribution': True,
#             'entropy': 0.0
#         }

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import config
from scipy.stats import entropy

# Thresholds for confidence and out-of-distribution detection
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for accepting predictions
ENTROPY_THRESHOLD = 1.2     # Entropy threshold for out-of-distribution detection

from tensorflow.keras.models import load_model

def load_trained_model(model_path="models/model_inception.h5"):
    """
    Load the trained model from the specified path.
    
    Args:
        model_path (str): Path to the saved model file.
        
    Returns:
        Keras model if successful, None otherwise.
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model
model = load_trained_model()
    
def calculate_prediction_entropy(probabilities):
    """
    Calculate entropy of prediction probabilities.
    Higher entropy indicates uncertainty (possibly OOD).
    
    Args:
        probabilities: Array of class probabilities
        
    Returns:
        Entropy value
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    probabilities = np.clip(probabilities, epsilon, 1.0)
    return -np.sum(probabilities * np.log(probabilities))
    
def predict_defect(model, img_processed):
    """
    Predicts the defect class for a given input image.
    Applies confidence-based filtering and out-of-distribution detection.
    
    Args:
        model: Loaded Keras model
        img_processed: Preprocessed image tensor (batch of 1)
        
    Returns:
        Dictionary with prediction results and confidence metrics
    """
    if model is None:
        return {
            'defect_type': 'Unknown - Model not loaded',
            'confidence': 0.0,
            'is_out_of_distribution': True,
            'entropy': 0.0
        }
    
    try:
        # Get prediction probabilities
        pred_probs = model.predict(img_processed, verbose=0)
        
        # Calculate metrics
        predicted_class_index = np.argmax(pred_probs[0])
        confidence = float(pred_probs[0][predicted_class_index])
        pred_entropy = calculate_prediction_entropy(pred_probs[0])
        
        print(f"Raw predictions: {pred_probs[0]}")
        print(f"Predicted class index: {predicted_class_index}, Confidence: {confidence:.4f}, Entropy: {pred_entropy:.4f}")
        
        # Determine if prediction is likely out-of-distribution
        is_out_of_distribution = (confidence < CONFIDENCE_THRESHOLD) or (pred_entropy > ENTROPY_THRESHOLD)
        
        if is_out_of_distribution:
            if confidence < CONFIDENCE_THRESHOLD:
                defect_type = "Uncertain - Low confidence"
            else:
                defect_type = "Uncertain - Possibly out-of-distribution"
        else:
            defect_type = config.DEFECT_CLASSES[predicted_class_index]
        
        return {
            'defect_type': defect_type,
            'confidence': confidence,
            'is_out_of_distribution': is_out_of_distribution,
            'entropy': pred_entropy,
            'raw_predictions': pred_probs[0].tolist(),
            'class_index': int(predicted_class_index)
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'defect_type': f'Error during prediction: {str(e)}',
            'confidence': 0.0,
            'is_out_of_distribution': True,
            'entropy': 0.0
        }