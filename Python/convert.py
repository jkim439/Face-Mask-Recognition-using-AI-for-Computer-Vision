# Title: Face Mask Detector
# Author: Junghwan Kim
# Date: April 8, 2021

# convert.py: Convert trained model to TenserFlow Lite model

import tensorflow as tf

print(tf.__version__)
print(help(tf.lite.TFLiteConverter))

saved_model_dir = "/Users/jkim/Classes/MSC/Project/TensorFlow/my_model"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(
    saved_model_dir
)  # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
