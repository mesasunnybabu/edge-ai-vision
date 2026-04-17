import tensorflow as tf
import time
import os
import numpy as np

print("--- Starting Model Optimization (Quantization) ---")

# 1. Load the original base model
print("Loading base model...")
model = tf.keras.models.load_model("base_model.keras")

# 2. Set up the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Apply Post-Training Quantization
# This flag tells TF Lite to shrink 32-bit floats down to 8-bit integers where possible!
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("Converting and applying Quantization...")
quantized_tflite_model = converter.convert()

# 4. Save the optimized model
quantized_path = "quantized_model.tflite"
with open(quantized_path, "wb") as f:
    f.write(quantized_tflite_model)

# 5. Measure the new optimized file size
size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
print(f"\nQuantized Model Size: {size_mb:.2f} MB")

# 6. Measure Optimized Inference Time
print("\nSetting up Interpreter for Quantized Model...")
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load test data
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = (x_test / 255.0).astype(np.float32)

print("Measuring Quantized inference time (1000 images)...")
# Warmup
interpreter.set_tensor(input_details[0]['index'], x_test[0:1])
interpreter.invoke() 

# Actual timing
start_time = time.time()
for i in range(1000):
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1])
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
end_time = time.time()

time_per_image_ms = ((end_time - start_time) / 1000) * 1000
print(f"Quantized Inference Time: {time_per_image_ms:.4f} ms / image")
print("--- Optimization Complete ---")