import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import IMG_HEIGHT, IMG_WIDTH 
import os
import urllib.request
from io import BytesIO
from PIL import Image

def is_url(path):
    """Check if the path is a URL."""
    return path.startswith(('http://', 'https://'))

def load_image(image_path):
    """
    Load images from various sources (local file, URL) and formats.
    
    Args:
        image_path: Path or URL to the image
        
    Returns:
        Loaded image as RGB numpy array
    """
    try:
        if is_url(image_path):
            # Handle URL
            with urllib.request.urlopen(image_path) as response:
                image_data = response.read()
            pil_image = Image.open(BytesIO(image_data))
            image = np.array(pil_image.convert("RGB"))
        elif os.path.exists(image_path):
            # Handle local file
            if image_path.lower().endswith(('.tif', '.tiff')):
                try:
                    import tifffile
                    image = tifffile.imread(image_path)
                    # Handle TIFF specifics (different bit depths, etc.)
                    if np.issubdtype(image.dtype, np.floating):
                        image = (np.clip(image * 255, 0, 255)).astype(np.uint8)
                    elif image.dtype == np.uint16:
                        image = (image / 256).astype(np.uint8)
                except ImportError:
                    # Fallback if tifffile not available
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image.convert("RGB"))
            else:
                # Regular image formats
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Ensure image is RGB
        if len(image.shape) == 2:  # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA to RGB
            image = image[:, :, :3]
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Single channel to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess image for model input.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure image is in correct format
    image = safe_image_conversion(image)

    # Resize to model's input shape
    image = cv2.resize(image, (224, 224))  # Fixed to match ResNet input size

    # Normalize pixel values (simple 0-1 normalization for ResNet50V2)
    normalized = image.astype(np.float32) / 255.0
    
    return normalized

def safe_image_conversion(image):
    """
    Safely convert image to uint8 format with proper scaling.
    """
    if image is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
        
    # Handle different image types and bit depths
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image * 255, 0, 255)).astype(np.uint8)
    elif image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Ensure image is RGB
    if len(image.shape) == 2:  # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA to RGB
        image = image[:, :, :3]
    elif image.shape[2] == 1:  # Single channel to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

def create_edge_mask(image, method='auto', margin_percent=10):
    """
    Create a mask to exclude edges from analysis.
    
    Args:
        image: Input image
        method: 'auto' for automatic detection, 'fixed' for fixed margin
        margin_percent: Percentage of image to exclude from edges for fixed method
        
    Returns:
        Binary mask (255 for regions to keep, 0 for regions to exclude)
    """
    # Ensure image is properly formatted
    image = safe_image_conversion(image)
    h, w = image.shape[:2]
    
    if method == 'fixed':
        # Simple fixed margin approach
        mask = np.ones((h, w), dtype=np.uint8) * 255
        margin_h = int(h * margin_percent / 100)
        margin_w = int(w * margin_percent / 100)
        mask[:margin_h, :] = 0
        mask[-margin_h:, :] = 0
        mask[:, :margin_w] = 0
        mask[:, -margin_w:] = 0
        return mask
    
    # Automatic detection approach
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if contours:
        # Find largest contour (assumed to be the solar cell)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw filled contour on mask
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Erode to exclude edges
        erode_kernel = np.ones((15, 15), np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=1)
    else:
        # If no contour found, use fixed margins
        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
        mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
    
    return mask

def apply_mask(image, mask):
    """
    Apply mask to an image.
    
    Args:
        image: Input image
        mask: Binary mask
        
    Returns:
        Masked image
    """
    # Ensure proper formatting
    image = safe_image_conversion(image)
    
    # Normalize mask to range 0-1
    mask = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask

    # Make mask compatible with RGB image
    if len(image.shape) == 3 and len(mask.shape) == 2:
        mask = np.stack([mask] * 3, axis=2)

    # Apply mask
    masked = (image * mask).astype(np.uint8)
    
    return masked

def process_image_for_prediction(image_path, apply_edge_masking=True):
    """
    Complete pipeline to process an image for prediction and analysis.

    Args:
        image_path: Path or URL to the image file
        apply_edge_masking: Whether to apply edge masking

    Returns:
        Dictionary with processed data for model input and visualization
    """
    # Load image from path or URL
    original_image = load_image(image_path)
    
    if original_image is None:
        print(f"Failed to load image: {image_path}")
        # Return a default black image to avoid crashes
        return {
            'original_image': np.zeros((224, 224, 3), dtype=np.uint8),
            'processed_image': np.zeros((224, 224, 3), dtype=np.uint8),
            'model_input': np.zeros((1, 224, 224, 3), dtype=np.float32)
        }
    
    # Resize to standard size
    processed_image = cv2.resize(original_image, (224, 224))
    
    # Apply masking if requested
    if apply_edge_masking:
        mask = create_edge_mask(processed_image, method='auto')
        masked_image = apply_mask(processed_image, mask)
    else:
        mask = np.ones((224, 224), dtype=np.uint8) * 255
        masked_image = processed_image.copy()
    
    # Create model input tensor
    model_input = np.expand_dims(preprocess_image(masked_image), axis=0)
    
    return {
        'original_image': original_image,
        'processed_image': processed_image,
        'mask': mask,
        'masked_image': masked_image,
        'model_input': model_input
    }