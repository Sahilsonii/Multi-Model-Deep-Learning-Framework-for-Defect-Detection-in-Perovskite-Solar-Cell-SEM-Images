# import os
# import numpy as np
# import tifffile as tiff
# import cv2
# from config import IMG_HEIGHT, IMG_WIDTH
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# def load_tiff_image(file_path):
#     """Load a TIFF image and convert to a format suitable for processing."""
#     try:
#         # Read TIFF file
#         img = tiff.imread('data\train\3D perovskite with pinholes\01-10.tif')
        
#         # Handle different channel formats
#         if len(img.shape) == 2:  # Grayscale
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         elif len(img.shape) == 3:
#             if img.shape[2] > 3:  # More than 3 channels
#                 img = img[:, :, :3]  # Keep only first 3 channels
        
#         # Normalize pixel values to 0-1 range if they're not already
#         if img.dtype == np.uint16:
#             img = img.astype(np.float32) / 65535.0
#         elif img.dtype == np.uint8:
#             img = img.astype(np.float32) / 255.0
        
#         return img
#     except Exception as e:
#         print(f"Error loading image {file_path}: {e}")
#         return None

# def organize_dataset(data_dir, output_dir, validation_split=0.15, test_split=0.15):
#     """
#     Organize the dataset into train, validation, and test sets.
    
#     Args:
#         data_dir: Directory containing class folders with images
#         output_dir: Directory where to save organized dataset
#         validation_split: Fraction of data to use for validation
#         test_split: Fraction of data to use for testing
#     """
#     # Create output directories
#     os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
#     # Process each class folder
#     class_names = [folder for folder in os.listdir(data_dir) 
#                   if os.path.isdir(os.path.join(data_dir, folder))]
    
#     for class_name in class_names:
#         print(f"Processing class: {class_name}")
        
#         # Create class directories in each split
#         os.makedirs(os.path.join(output_dir, 'train', class_name), exist_ok=True)
#         os.makedirs(os.path.join(output_dir, 'validation', class_name), exist_ok=True)
#         os.makedirs(os.path.join(output_dir, 'test', class_name), exist_ok=True)
        
#         # Get all image files in the class directory
#         class_dir = os.path.join(data_dir, class_name)
#         image_files = [f for f in os.listdir(class_dir) 
#                       if f.lower().endswith(('.tif', '.tiff'))]
        
#         # Split the data
#         train_files, test_files = train_test_split(
#             image_files, test_size=test_split, random_state=42)
        
#         train_files, val_files = train_test_split(
#             train_files, test_size=validation_split/(1-test_split), random_state=42)
        
#         # Copy files to their respective directories
#         for file_list, split_name in [(train_files, 'train'), 
#                                      (val_files, 'validation'), 
#                                      (test_files, 'test')]:
#             for file_name in file_list:
#                 source = os.path.join(class_dir, file_name)
#                 destination = os.path.join(output_dir, split_name, class_name, file_name)
                
#                 # Load, preprocess and save the image
#                 img = load_tiff_image(source)
#                 if img is not None:
#                     # Convert to 8-bit for storage efficiency
#                     img_8bit = (img * 255).astype(np.uint8)
#                     cv2.imwrite(destination.replace('.tif', '.jpg').replace('.tiff', '.jpg'), 
#                                cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR))

# def check_dataset(data_dir):
#     """
#     Print statistics about the dataset.
    
#     Args:
#         data_dir: Directory containing train, validation, test folders
#     """
#     for split in ['train', 'validation', 'test']:
#         print(f"\n{split.upper()} SET:")
#         split_dir = os.path.join(data_dir, split)
#         total_images = 0
        
#         for class_name in os.listdir(split_dir):
#             class_dir = os.path.join(split_dir, class_name)
#             if os.path.isdir(class_dir):
#                 num_images = len([f for f in os.listdir(class_dir) 
#                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
#                 print(f"  - {class_name}: {num_images} images")
#                 total_images += num_images
        
#         print(f"  Total: {total_images} images")

# def create_edge_mask(image):
#     """
#     Create a mask that excludes the edges of the solar cell.
    
#     Args:
#         image: Input RGB image
    
#     Returns:
#         Binary mask (255 for active area, 0 for edges/background)
#     """
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
#     # Convert to binary using Otsu's thresholding
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
#     # Perform morphological operations to clean up the mask
#     kernel = np.ones((5, 5), np.uint8)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
#     # Find contours
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Create empty mask
#     mask = np.zeros_like(gray)
    
#     if contours:
#         # Find largest contour (assumed to be the solar cell)
#         cell_contour = max(contours, key=cv2.contourArea)
        
#         # Draw filled contour on mask
#         cv2.drawContours(mask, [cell_contour], 0, 255, -1)
        
#         # Erode to exclude edges
#         erode_kernel = np.ones((15, 15), np.uint8)
#         mask = cv2.erode(mask, erode_kernel, iterations=1)
#     else:
#         # If no contour found, exclude outer 10% of image
#         h, w = mask.shape
#         margin_h, margin_w = int(h * 0.1), int(w * 0.1)
#         mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
    
#     return mask

# def analyze_pinholes(image, mask):
#     """
#     Detect and analyze pinholes in the masked solar cell image.
    
#     Args:
#         image: RGB image
#         mask: Binary mask of active area
    
#     Returns:
#         Dictionary containing pinhole analysis results and visualizations
#     """
#     # Apply mask to image
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    
#     # Apply contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)
    
#     # Blur to reduce noise
#     blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
#     # Use adaptive thresholding to identify potential pinholes
#     thresh = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY_INV, 11, 2
#     )
    
#     # Apply mask to threshold to eliminate detections outside active area
#     thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
#     # Find contours of potential pinholes
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter and analyze pinholes
#     pinholes = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)
        
#         # Calculate circularity
#         circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
#         # Filter based on size and shape
#         if 10 < area < 500 and circularity > 0.6:
#             x, y, w, h = cv2.boundingRect(contour)
            
#             pinholes.append({
#                 'contour': contour,
#                 'position': (x + w//2, y + h//2),
#                 'area': area,
#                 'circularity': circularity
#             })
    
#     # Create visualization of detected pinholes
#     visualization = image.copy()
#     cv2.drawContours(visualization, [p['contour'] for p in pinholes], -1, (255, 0, 0), 2)
    
#     # Calculate statistics
#     active_area = cv2.countNonZero(mask)
#     pinhole_count = len(pinholes)
#     avg_pinhole_size = np.mean([p['area'] for p in pinholes]) if pinholes else 0
#     total_pinhole_area = sum(p['area'] for p in pinholes)
#     defect_ratio = (total_pinhole_area / active_area * 100) if active_area > 0 else 0
    
#     # Return results
#     return {
#         'pinhole_count': pinhole_count,
#         'avg_pinhole_size': avg_pinhole_size,
#         'defect_ratio': defect_ratio,
#         'active_area': active_area,
#         'binary_mask': thresh,
#         'visualization': visualization
#     }

# def analyze_solar_cell(image_path):
#     """
#     Perform complete analysis of a solar cell image.
    
#     Args:
#         image_path: Path to the image file
    
#     Returns:
#         Dictionary containing analysis results
#     """
#     # Load image (handles both TIFF and other formats)
#     if image_path.lower().endswith(('.tif', '.tiff')):
#         original_image = load_tiff_image(image_path)
#         # Convert from float to uint8 for OpenCV
#         if original_image.dtype == np.float32 or original_image.dtype == np.float64:
#             original_image = (original_image * 255).astype(np.uint8)
#     else:
#         original_image = cv2.imread(image_path)
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
#     # Create edge mask
#     mask = create_edge_mask(original_image)
    
#     # Analyze pinholes
#     pinhole_analysis = analyze_pinholes(original_image, mask)
    
#     # Prepare image for model (resized, masked)
#     masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    
#     # Create overlays for visualization
#     # Overlay 1: Mask on original image
#     mask_overlay = original_image.copy()
#     green_mask = np.zeros_like(original_image)
#     green_mask[mask > 0] = [0, 255, 0]
#     mask_overlay = cv2.addWeighted(mask_overlay, 1, green_mask, 0.3, 0)
    
#     return {
#         'original_image': original_image,
#         'mask': mask,
#         'masked_image': masked_image,
#         'pinhole_analysis': pinhole_analysis,
#         'mask_overlay': mask_overlay
#     }        

# # Add support for BMP files in image processing
# def process_image_for_prediction(filepath, apply_edge_masking=True):
#     """
#     Process image for model prediction with BMP support.
    
#     Args:
#         filepath: Path to the uploaded image
#         apply_edge_masking: Whether to apply edge masking
    
#     Returns:
#         Dictionary with processed image data
#     """
#     try:
#         # Read image with OpenCV, which supports multiple formats including BMP
#         img = cv2.imread(filepath)
        
#         # Convert BGR to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Resize image
#         img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        
#         # Optional edge masking
#         if apply_edge_masking:
#             mask = create_edge_mask(img_resized)
#             img_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        
#         # Normalize
#         img_normalized = img_resized.astype(np.float32) / 255.0
        
#         # Prepare for model input
#         model_input = np.expand_dims(img_normalized, axis=0)
        
#         return {
#             'original_image': img,
#             'processed_image': img_resized,
#             'model_input': model_input
#         }
#     except Exception as e:
#         print(f"Image processing error: {e}")
#         return None

import os
import numpy as np
import tifffile as tiff
import cv2
from config import IMG_HEIGHT, IMG_WIDTH
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_tiff_image(file_path):
    """Load a TIFF image and convert to a format suitable for processing."""
    try:
        # Read TIFF file
        img = tiff.imread('data\train\3D perovskite with pinholes\01-10.tif')
        
        # Handle different channel formats
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] > 3:  # More than 3 channels
                img = img[:, :, :3]  # Keep only first 3 channels
        
        # Normalize pixel values to 0-1 range if they're not already
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def organize_dataset(data_dir, output_dir, validation_split=0.15, test_split=0.15):
    """
    Organize the dataset into train, validation, and test sets.
    
    Args:
        data_dir: Directory containing class folders with images
        output_dir: Directory where to save organized dataset
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Process each class folder
    class_names = [folder for folder in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, folder))]
    
    for class_name in class_names:
        print(f"Processing class: {class_name}")
        
        # Create class directories in each split
        os.makedirs(os.path.join(output_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'validation', class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', class_name), exist_ok=True)
        
        # Get all image files in the class directory
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.tif', '.tiff'))]
        
        # Split the data
        train_files, test_files = train_test_split(
            image_files, test_size=test_split, random_state=42)
        
        train_files, val_files = train_test_split(
            train_files, test_size=validation_split/(1-test_split), random_state=42)
        
        # Copy files to their respective directories
        for file_list, split_name in [(train_files, 'train'), 
                                     (val_files, 'validation'), 
                                     (test_files, 'test')]:
            for file_name in file_list:
                source = os.path.join(class_dir, file_name)
                destination = os.path.join(output_dir, split_name, class_name, file_name)
                
                # Load, preprocess and save the image
                img = load_tiff_image(source)
                if img is not None:
                    # Convert to 8-bit for storage efficiency
                    img_8bit = (img * 255).astype(np.uint8)
                    cv2.imwrite(destination.replace('.tif', '.jpg').replace('.tiff', '.jpg'), 
                               cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR))

def check_dataset(data_dir):
    """
    Print statistics about the dataset.
    
    Args:
        data_dir: Directory containing train, validation, test folders
    """
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} SET:")
        split_dir = os.path.join(data_dir, split)
        total_images = 0
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                num_images = len([f for f in os.listdir(class_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
                print(f"  - {class_name}: {num_images} images")
                total_images += num_images
        
        print(f"  Total: {total_images} images")

def create_edge_mask(image):
    """
    Create a mask that excludes the edges of the solar cell.
    
    Args:
        image: Input RGB image
    
    Returns:
        Binary mask (255 for active area, 0 for edges/background)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert to binary using Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask
    mask = np.zeros_like(gray)
    
    if contours:
        # Find largest contour (assumed to be the solar cell)
        cell_contour = max(contours, key=cv2.contourArea)
        
        # Draw filled contour on mask
        cv2.drawContours(mask, [cell_contour], 0, 255, -1)
        
        # Erode to exclude edges
        erode_kernel = np.ones((15, 15), np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=1)
    else:
        # If no contour found, exclude outer 10% of image
        h, w = mask.shape
        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
        mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
    
    return mask

def analyze_pinholes(image, mask):
    """
    Detect and analyze pinholes in the masked solar cell image.
    
    Args:
        image: RGB image
        mask: Binary mask of active area
    
    Returns:
        Dictionary containing pinhole analysis results and visualizations
    """
    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Use adaptive thresholding to identify potential pinholes
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply mask to threshold to eliminate detections outside active area
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    # Find contours of potential pinholes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and analyze pinholes
    pinholes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Filter based on size and shape
        if 10 < area < 500 and circularity > 0.6:
            x, y, w, h = cv2.boundingRect(contour)
            
            pinholes.append({
                'contour': contour,
                'position': (x + w//2, y + h//2),
                'area': area,
                'circularity': circularity
            })
    
    # Create visualization of detected pinholes
    visualization = image.copy()
    cv2.drawContours(visualization, [p['contour'] for p in pinholes], -1, (255, 0, 0), 2)
    
    # Calculate statistics
    active_area = cv2.countNonZero(mask)
    pinhole_count = len(pinholes)
    avg_pinhole_size = np.mean([p['area'] for p in pinholes]) if pinholes else 0
    total_pinhole_area = sum(p['area'] for p in pinholes)
    defect_ratio = (total_pinhole_area / active_area * 100) if active_area > 0 else 0
    
    # Return results
    return {
        'pinhole_count': pinhole_count,
        'avg_pinhole_size': avg_pinhole_size,
        'defect_ratio': defect_ratio,
        'active_area': active_area,
        'binary_mask': thresh,
        'visualization': visualization
    }

def analyze_solar_cell(image_path):
    """
    Perform complete analysis of a solar cell image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary containing analysis results
    """
    # Load image (handles both TIFF and other formats)
    if image_path.lower().endswith(('.tif', '.tiff')):
        original_image = load_tiff_image(image_path)
        # Convert from float to uint8 for OpenCV
        if original_image.dtype == np.float32 or original_image.dtype == np.float64:
            original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create edge mask
    mask = create_edge_mask(original_image)
    
    # Analyze pinholes
    pinhole_analysis = analyze_pinholes(original_image, mask)
    
    # Prepare image for model (resized, masked)
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    
    # Create overlays for visualization
    # Overlay 1: Mask on original image
    mask_overlay = original_image.copy()
    green_mask = np.zeros_like(original_image)
    green_mask[mask > 0] = [0, 255, 0]
    mask_overlay = cv2.addWeighted(mask_overlay, 1, green_mask, 0.3, 0)
    
    return {
        'original_image': original_image,
        'mask': mask,
        'masked_image': masked_image,
        'pinhole_analysis': pinhole_analysis,
        'mask_overlay': mask_overlay
    }        

# Add support for BMP files in image processing
def process_image_for_prediction(filepath, apply_edge_masking=True):
    """
    Process image for model prediction with BMP support.
    
    Args:
        filepath: Path to the uploaded image
        apply_edge_masking: Whether to apply edge masking
    
    Returns:
        Dictionary with processed image data
    """
    try:
        # Read image with OpenCV, which supports multiple formats including BMP
        img = cv2.imread(filepath)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        
        # Optional edge masking
        if apply_edge_masking:
            mask = create_edge_mask(img_resized)
            img_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Prepare for model input
        model_input = np.expand_dims(img_normalized, axis=0)
        
        return {
            'original_image': img,
            'processed_image': img_resized,
            'model_input': model_input
        }
    except Exception as e:
        print(f"Image processing error: {e}")
        return None