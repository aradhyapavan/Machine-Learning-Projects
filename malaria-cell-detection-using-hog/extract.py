import os
import zipfile

# Define paths
zip_path = 'cell_images.zip'
extract_folder = 'cell_images'

# Check if the zip file exists
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"Zip file '{zip_path}' not found. Please make sure it exists in the working directory.")

# Create the folder if it doesn't exist
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)
    print(f"Created folder: {extract_folder}")
else:
    print(f"Folder already exists: {extract_folder}")

# Extract zip file using Python's standard library 'zipfile'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
    print(f"Extracted all files from {zip_path} to {extract_folder}") 