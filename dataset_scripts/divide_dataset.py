import os
import shutil
import random

dataset_path = ""
output_dir = ""
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

images = [f for f in os.listdir(dataset_path) if f.endswith(".png") and not f.startswith(".") and os.path.isfile(os.path.join(dataset_path, f))]
random.shuffle(images)
total = len(images)
train_count = int(0.8 * total)
val_count = int(0.1 * total)

train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]


def copy_data(image_list, dest_dir):
    for img in image_list:
        img_path = os.path.join(dataset_path, img)
        img_name = os.path.splitext(img)[0]
        folder_path = os.path.join(dataset_path, img_name)

        shutil.copy(img_path, os.path.join(dest_dir, img))

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.copytree(folder_path, os.path.join(dest_dir, img_name), dirs_exist_ok=True)


copy_data(train_images, train_dir)
copy_data(val_images, val_dir)
copy_data(test_images, test_dir)

print("Dataset split completed.")
