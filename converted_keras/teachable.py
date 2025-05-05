from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Print the current working directory for debugging purposes
print("Current Directory:", os.getcwd())

# Load the model with the full path
model = load_model(r"C:\Users\User\OneDrive - University of Nottingham Malaysia\Documents\National ai Competition\converted_keras\keras_model.h5", compile=False)

# Load the labels with the full path
class_names = open(r"C:\Users\User\OneDrive - University of Nottingham Malaysia\Documents\National ai Competition\converted_keras\labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Define paths to test folders
test_folders = [
    r"C:\Users\User\OneDrive - University of Nottingham Malaysia\Documents\National ai Competition\DataSet\Test\kuih_ketayap",
    r"C:\Users\User\OneDrive - University of Nottingham Malaysia\Documents\National ai Competition\DataSet\Test\kuih_ubi_kayu"
]

# Loop through both folders
for folder in test_folders:
    print(f"\nClassifying images in folder: {folder}")
    
    for filename in os.listdir(folder):
        # Construct full image path
        image_path = os.path.join(folder, filename)
        
        if image_path.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            try:
                # Open the image and convert to RGB
                image = Image.open(image_path).convert("RGB")
                
                # Resize and crop the image to 224x224
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                
                # Convert image to numpy array
                image_array = np.asarray(image)
                
                # Normalize the image
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                
                # Load the image into the data array
                data[0] = normalized_image_array
                
                # Predict using the model
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                
                # Print the results for each image
                print(f"Class: {class_name.strip()} | Confidence Score: {confidence_score}")
            
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
