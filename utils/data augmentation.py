import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configs
input_dir = r"C:\Users\ASUS\Desktop\New folder (2)\aug classess"
output_dir = "augmented_only_dir"
target_size = (1024, 768)
target_count = 500  # total desired images per class
max_augs_per_image = 10

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect'
)

# Helper function to load supported image formats
def load_image(path):
    try:
        return Image.open(path).convert("RGB").resize(target_size)
    except:
        return None

# Prepare output folders
os.makedirs(output_dir, exist_ok=True)

# Track augmentation count per class
aug_counts = {}

class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

print("Augmenting Classes:")
for class_name in tqdm(class_dirs):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    image_files = [f for f in os.listdir(class_input_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    current_count = len(image_files)
    images_needed = target_count - current_count
    if current_count == 0 or images_needed <= 0:
        aug_counts[class_name] = 0
        continue

    augmentations_per_image = min((images_needed + current_count - 1) // current_count, max_augs_per_image)
    augmented_total = 0

    for img_name in image_files:
        img_path = os.path.join(class_input_path, img_name)
        img = load_image(img_path)
        if img is None:
            continue

        x = np.expand_dims(np.array(img), 0)

        gen = datagen.flow(x, batch_size=1)
        base_name = os.path.splitext(img_name)[0]

        for i in range(augmentations_per_image):
            if augmented_total >= images_needed:
                break
            aug_img = next(gen)[0].astype(np.uint8)
            aug_img_pil = Image.fromarray(aug_img)
            save_path = os.path.join(class_output_path, f"{base_name}_aug{i+1}.jpg")
            aug_img_pil.save(save_path)
            augmented_total += 1

    aug_counts[class_name] = augmented_total

# Plot chart
plt.figure(figsize=(12, 6))
classes = list(aug_counts.keys())
counts = list(aug_counts.values())

bars = plt.bar(classes, counts, color='skyblue')
plt.title("Number of Augmented Images per Class")
plt.xlabel("Class Names")
plt.ylabel("Number of Augmented Images")
plt.xticks(rotation=20, ha='right')

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, str(yval), ha='center', va='bottom')

# Save chart as image
plt.tight_layout()
plt.savefig("augmented_image_distribution.png")
plt.show()