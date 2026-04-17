import tensorflow as tf
import time
import os
import numpy as np

print("--- Starting TFLite Conversion ---")

# 1. Load the base model we just saved
print("Loading base model...")
model = tf.keras.models.load_model("base_model.keras")

# 2. Convert the model to TFLite format
print("Converting to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save the TFLite model to disk
tflite_path = "unoptimized_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

# 4. Measure the new file size
size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
print(f"\nNew TFLite File Size: {size_mb:.2f} MB")

# 5. Measure TFLite Inference Time
print("\nSetting up TFLite Interpreter to test speed...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load a bit of test data for the speed test
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = (x_test / 255.0).astype(np.float32) # TFLite strict requirement: must be float32

print("Measuring TFLite inference time (1000 images)...")
# Warmup run
interpreter.set_tensor(input_details[0]['index'], x_test[0:1])
interpreter.invoke()

# Actual timing
start_time = time.time()
for i in range(1000):
    # TFLite requires we feed images one by one
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1])
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
end_time = time.time()

time_per_image_ms = ((end_time - start_time) / 1000) * 1000
print(f"TFLite Inference Time: {time_per_image_ms:.4f} ms / image")
print("--- Conversion Complete ---")