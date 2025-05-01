import os
import tensorflow as tf
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_DIR, TRAIN_DIR, IMG_HEIGHT, IMG_WIDTH, VALIDATION_DIR, BATCH_SIZE, TEST_DIR

def load_image_safely(file_path):
    """
    Load an image from various formats with robust error handling.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Loaded image or None if failed
    """
    try:
        # Extract file extension
        ext = os.path.splitext(file_path.lower())[1]
        
        if ext in ['.tif', '.tiff']:
            import tifffile as tiff
            img = tiff.imread(file_path)
            
            # Convert uint16 to uint8
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            
            # Convert float64/float32 to uint8
            elif np.issubdtype(img.dtype, np.floating):
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
        
        elif ext in ['.bmp']:
            # Special handling for BMP
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        else:
            # Regular image formats (jpg, png)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Handle color format
        if img is not None:
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] > 3:  # RGBA or more channels
                img = img[:, :, :3]
            
            # Final check for floating point data
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
                
            return img
    
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
    
    return None

def organize_dataset(source_folder, defect_types, split_ratios=(0.80, 0.10, 0.10)):# yeh hundreds percent split use hoty 80% train, 10% val or test ki or test
    """
    Organize dataset with improved image handling.
    
    Args:
        source_folder: Path to source images
        defect_types: List of defect categories
        split_ratios: Train/Val/Test split ratios
    """
    print(f"Organizing dataset from {source_folder}")
    
    # Create directories
    for split in ['train', 'validation', 'test']:
        for defect in defect_types:
            os.makedirs(os.path.join(DATA_DIR, split, defect), exist_ok=True)

    # Process each defect type
    for defect in defect_types:
        defect_files = []
        search_term = defect.lower().replace('-', '_').replace(' ', '_')
        
        # Find image files
        for file in os.listdir(source_folder):
            file_lower = file.lower()
            if file_lower.endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')):
                if search_term in file_lower:
                    defect_files.append(os.path.join(source_folder, file))

        if not defect_files:
            print(f"No files found for defect type: {defect}")
            continue

        # Split data
        train_files, test_files = train_test_split(defect_files, test_size=split_ratios[2], random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=split_ratios[1]/(split_ratios[0]+split_ratios[1]), random_state=42)

        # Process splits
        for file_group, dest_dir, split_name in [
            (train_files, os.path.join(DATA_DIR, 'train', defect), "Train"),
            (val_files, os.path.join(DATA_DIR, 'validation', defect), "Validation"),
            (test_files, os.path.join(DATA_DIR, 'test', defect), "Test")
        ]:
            processed_count = 0
            for file in file_group:
                # Determine output filename, preserve original extension
                base_name = os.path.basename(file)
                file_ext = os.path.splitext(base_name)[1].lower()
                
                # Force JPG for compatibility with ImageDataGenerator
                dest_file = os.path.join(dest_dir, os.path.splitext(base_name)[0] + '.jpg')
                
                # Load with robust error handling
                img = load_image_safely(file)
                if img is not None:
                    # Save as JPG for best compatibility
                    cv2.imwrite(dest_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    processed_count += 1
                else:
                    print(f"Failed to process {file}")
            
            print(f"  - {split_name}: Processed {processed_count}/{len(file_group)} images for {defect}")

def create_data_generators():
    """Create data generators for training, validation and testing without augmentation."""
    print("Creating data generators...")
    
    # Print actual file counts
    for split, dir_path in [("Training", TRAIN_DIR), ("Validation", VALIDATION_DIR), ("Test", TEST_DIR)]:
        print(f"\n{split} directory contents:")
        if os.path.exists(dir_path):
            for class_name in sorted(os.listdir(dir_path)):
                class_dir = os.path.join(dir_path, class_name)
                if os.path.isdir(class_dir):
                    file_count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  - {class_name}: {file_count} images")
    
    # Create data generators
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, valid_generator, test_generator

def visualize_samples(data_generator, num_samples=5):
    """Visualize sample images from a data generator."""
    images, labels = next(data_generator)
    class_names = list(data_generator.class_indices.keys())

    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i])
        class_idx = np.argmax(labels[i])
        plt.title(f"Class: {class_names[class_idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_dataset_distribution():
    """Analyze and visualize the distribution of images across classes and splits."""
    print("\nAnalyzing dataset distribution...")
    splits = ['train', 'validation', 'test']
    class_counts = {}

    for split in splits:
        class_counts[split] = {}
        split_dir = os.path.join(DATA_DIR, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist!")
            continue
            
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                # Count all supported image file formats
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))])
                class_counts[split][class_name] = count

    print("Dataset Distribution:")
    for split in splits:
        print(f"\n{split.capitalize()} Set:")
        if split in class_counts:
            for class_name, count in class_counts[split].items():
                print(f"  - {class_name}: {count} images")
        else:
            print("  No data")

    # Create visualization if possible
    if all(split in class_counts for split in splits):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_counts['train'].keys()))
        width = 0.25

        rects1 = ax.bar(x - width, class_counts['train'].values(), width, label='Train')
        rects2 = ax.bar(x, class_counts['validation'].values(), width, label='Validation')
        rects3 = ax.bar(x + width, class_counts['test'].values(), width, label='Test')

        ax.set_ylabel('Number of Images')
        ax.set_title('Dataset Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(class_counts['train'].keys(), rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    defect_types = [
        "3D perovskite",
        "3D-2D mixed perovskite",
        "3D perovskite with PbI2 excess",
        "3D perovskite with pinholes",
        "3D-2D mixed perovskite with pinholes"
    ]

    # Uncomment this line to organize dataset
    organize_dataset("./dataset", defect_types)  # Update path to your raw images

    # Create data generators after organizing
    train_gen, valid_gen, test_gen = create_data_generators()
    
    # Analyze the dataset distribution
    analyze_dataset_distribution()
    
    # Visualize sample images
    print("\nSample training images:")
    visualize_samples(train_gen)