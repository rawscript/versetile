from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

app = Flask(__name__)

# Load the model.
model = tf.keras.models.load_model('colorizer.h5')

@app.route('/colorize', methods=['POST'])
def colorize():
    # Get the image file from the request.
    image_file = request.files['image']

    # Convert the image to grayscale.
    image = Image.open(image_file)
    image = np.array(image)
    image = tf.image.rgb_to_grayscale(image)

    # Normalize the image.
    image = image / 255.0

    # Colorize the image.
    colored_image = model.predict(image)

    # Convert the colorized image to a byte string.
    colored_image_bytes = tf.image.encode_png(colored_image)

    # Return the colorized image as a JSON response.
    return jsonify({'image': colored_image_bytes.numpy().tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
