# Title: Face Mask Detector
# Author: Junghwan Kim
# Date: April 8, 2021

# test.py: Test trained model using test image

import numpy as np
import tensorflow as tf
from tensorflow import keras

path = "/Users/jkim/Classes/MSC/Project/TensorFlow/data"

model = keras.models.load_model("/Users/jkim/Classes/MSC/Project/TensorFlow/my_model")

img_height = 180
img_width = 180

img = keras.preprocessing.image.load_img(
    "/Users/jkim/Classes/MSC/Project/TensorFlow/test/1.jpg",
    target_size=(img_height, img_width),
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
class_names = ["with_mask", "without_mask"]

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
    )
)
