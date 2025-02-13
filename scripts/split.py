import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Define paths
base_dir = r"C:\Users\amark\yolo_dataset\yolo_waste_packaging_dataset"
image_dir = os.path.join(base_dir, "Images", "train")
label_dir = os.path.join(base_dir, "Labels", "train")

output_base = r"C:\Users\amark\yolo_dataset"
output_dirs = {
    "train": {"images": os.path.join(output_base, "images", "train"),
               "labels": os.path.join(output_base, "labels", "train")},
    "val": {"images": os.path.join(output_base, "images", "val"),
             "labels": os.path.join(output_base, "labels", "val")},
    "test": {"images": os.path.join(output_base, "images", "test"),
              "labels": os.path.join(output_base, "labels", "test")}
}

# Create output directories if they do not exist
for split in output_dirs:
    os.makedirs(output_dirs[split]["images"], exist_ok=True)
    os.makedirs(output_dirs[split]["labels"], exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)

# Split ratios
train_ratio = 0.7
val_ratio = 0.2

num_images = len(image_files)
num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)

train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

# Function to move files
def move_files(file_list, split):
    for file_name in file_list:
        # Move image file
        src_image = os.path.join(image_dir, file_name)
        dest_image = os.path.join(output_dirs[split]["images"], file_name)
        shutil.move(src_image, dest_image)
        
        # Move corresponding label file
        label_file = os.path.splitext(file_name)[0] + ".txt"
        src_label = os.path.join(label_dir, label_file)
        dest_label = os.path.join(output_dirs[split]["labels"], label_file)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dest_label)

# Move files to respective folders
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split complete.")
