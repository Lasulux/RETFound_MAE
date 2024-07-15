import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy

def get_first_image_from_each_directory(root_dir):
    image_paths = []

    for class_dir in os.listdir(root_dir):
        full_class_dir = os.path.join(root_dir, class_dir)
        if os.path.isdir(full_class_dir):
            first_image = sorted(os.listdir(full_class_dir))[0]
            image_path = os.path.join(full_class_dir, first_image)
            image_paths.append(image_path)

    return image_paths

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    left_half = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=0, target_height=512, target_width=512)
    image = tf.image.resize(left_half, [224, 224])  # Resize to a fixed size, e.g., 224x224
    image = image / 255.0  # Normalize to [0,1] range
    return image

def create_dataset(image_paths):
    image_paths = tf.convert_to_tensor(image_paths)

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda path: load_and_preprocess_image(path))
    return dataset

# Define the root directory of your dataset
root_dir = "./data/OIMHS dataset/Images"

# Get the first image from each directory
image_paths = get_first_image_from_each_directory(root_dir)

# Create the dataset
dataset = create_dataset(image_paths)
print("a")
# for image in dataset:
    # print(image.shape)
