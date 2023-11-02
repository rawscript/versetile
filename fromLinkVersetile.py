import tensorflow as tf
import numpy as np
from PIL import Image
import requests

# Load the model.
model = tf.keras.models.load_model('colorizer.h5')

# Download the image.
url = 'https://unsplash.com/photos/greyscale-photo-of-classic-vehicle-on-ground-V09Io5ln-Qo'
response = requests.get(url)
image = Image.open(BytesIO(response.content))
image = np.array(image)

# Convert the image to grayscale.
image = tf.image.rgb_to_grayscale(image)

# Normalize the image.
image = image / 255.0

# Colorize the image.
colored_image = model.predict(image)

# Save the colored image.
tf.keras.preprocessing.image.save_img('colored_image.jpg', colored_image)
