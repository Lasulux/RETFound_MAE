import tensorflow as tf
import tfimm


# Create the base model
model_name = "vit_large_patch16_224"
base_model = tfimm.create_model(
    model_name,
    nb_classes=0  # this removes the final layer
)

# Load base_model weights (by_name=True because nb_classes=0 removes a layer)
model_path = "RETFound_MAE/RETFound_cfp_weights.h5"
model_path = "RETFound_oct_weights.h5"

base_model.load_weights(model_path, by_name=True, skip_mismatch=False)
base_model.trainable = False
# base_model.summary()

# Input layer
input_shape = (224, 224, 3)
inputs = tf.keras.Input(shape=input_shape)

# ViT layer
x = base_model(inputs)

# Add layers to match the original architecture leading up to the 'head' layer
# x = tf.keras.layers.LayerNormalization()(x)
# x = tf.keras.layers.Dense(1024, activation='relu')(x)
prediction = tf.keras.layers.Dense(1, name='head')(x)


# Define the model
model = tf.keras.Model(inputs=inputs, outputs=prediction)
model.summary()


#%% Test model on a single image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the image
image_path = "RETFound_MAE/data/IDRiD_data_one_image/train/anoDR/IDRiD_118.png"
image_path = "./data/IDRiD_data/train/anoDR/IDRiD_118.png"

# Define the target input size for your model
input_shape = (224, 224)

# Load the image
img = image.load_img(image_path, target_size=input_shape)

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand the dimensions of the image to match the input shape of the model
img_input = np.expand_dims(img_array, axis=0)

# Preprocess the input by normalizing pixel values
img_input = img_input / 255

# Perform inference
predictions = model.predict(img_input)

# Assuming 'predictions' contains your model's output, you can use it as needed
print("Predictions:", predictions)