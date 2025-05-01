# from PIL import Image
# import os

# # Folder path containing the images
# folder_path = 'images'

# # Supported input formats
# input_extensions = ('.tif', '.bmp')

# # Loop through all files in the folder
# for file_name in os.listdir(folder_path):
#     if file_name.lower().endswith(input_extensions):
#         file_path = os.path.join(folder_path, file_name)

#         try:
#             # Open the image
#             img = Image.open(file_path)

#             # Convert to RGB if needed
#             if img.mode in ("RGBA", "P"):
#                 img = img.convert("RGB")

#             # Generate JPG file name
#             new_file_name = os.path.splitext(file_name)[0] + '.jpg'
#             new_file_path = os.path.join(folder_path, new_file_name)

#             # Save as JPG
#             img.save(new_file_path, 'JPEG')
#             print(f"Converted: {file_name} → {new_file_name}")

#             # Remove the original file
#             os.remove(file_path)
#             print(f"Deleted original file: {file_name}")

#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")


from PIL import Image
import os

# Folder path containing the images
folder_path = 'images'
input_extensions = ('.tif', '.bmp')

# Get list of files to convert
files_to_convert = [f for f in os.listdir(folder_path) if f.lower().endswith(input_extensions)]

# Sort for consistent numbering (optional)
files_to_convert.sort()

# Start counter
counter = 554

for file_name in files_to_convert:
    file_path = os.path.join(folder_path, file_name)

    try:
        # Open the image
        img = Image.open(file_path)

        # Convert to RGB if needed
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Set the new numbered filename
        new_file_name = f"{counter}.jpg"
        new_file_path = os.path.join(folder_path, new_file_name)

        # Save as JPG
        img.save(new_file_path, 'JPEG')
        print(f"Converted: {file_name} → {new_file_name}")

        # Remove original file
        os.remove(file_path)
        print(f"Deleted original file: {file_name}")

        counter += 1

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

