# organize_data.py
from utils import organize_dataset, check_dataset

# Change these paths to match your setup
input_directory = "data/train/"  # Directory with your 3 class folders containing TIFF images
output_directory = "dataset"     # Where to save the organized dataset

# Organize the dataset
organize_dataset(input_directory, output_directory)

# Check the organized dataset
check_dataset(output_directory)