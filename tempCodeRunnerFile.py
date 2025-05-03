import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from PIL import Image

# --- Config ---
source_dir = "C:\Users\User\OneDrive - University of Nottingham Malaysia\Documents\National ai Competition\DataSet"
target_dir = "C:\Users\User\OneDrive - University of Nottingham Malaysia\Documents\National ai Competition\AugmentedData"
img_size = (180, 180)
augment_per_image = 5  # How many augmented versions to generate per image

# --- Data Augmentation Pipeline ---
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomCrop(height=img_size[0], width=img_size[1]),
    layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.3)),
    layers.Lambda(lambda x: tf.image.random_saturation(x, lower=0.7, upper=1.3)),
    layers.Lambda(lambda x: x + tf.random.uniform(tf.shape(x), minval=0, maxval=0.1)),  # Add noise
    layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0)),
])

# --- Load Dataset ---
dataset = image_dataset_from_directory(
    source_dir,
    labels='inferred',
    label_mode='int',
    batch_size=1,  # Process one image at a time for saving
    image_size=img_size,
    shuffle=False
)

# --- Ensure target folders exist ---
class_names = dataset.class_names
for class_name in class_names:
    os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

# --- Generate & Save Augmented Images ---
counter = 0
for batch, labels in dataset:
    img_tensor = batch[0]
    label = labels[0].numpy()
    class_name = class_names[label]
    
    for i in range(augment_per_image):
        aug_img = data_augmentation(tf.expand_dims(img_tensor, 0))[0].numpy()
        aug_img = (aug_img * 255).astype(np.uint8)
        img_pil = Image.fromarray(aug_img)
        
        filename = f"aug_{counter:05d}_{i+1}.jpg"
        save_path = os.path.join(target_dir, class_name, filename)
        img_pil.save(save_path)
        print(f"Saved {save_path}")
    
    counter += 1
