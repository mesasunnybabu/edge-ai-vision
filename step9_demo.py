import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import random

print("--- Launching Final Edge AI Demo ---")

# 1. Load the Quantized Model
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Load test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = (x_test / 255.0).astype(np.float32)

# 3. Pick 5 random images
random_indices = random.sample(range(len(x_test)), 5)

plt.figure(figsize=(12, 3))
plt.suptitle("Edge AI: 8-Bit Quantized Model Live Inference", fontsize=14, fontweight='bold')

for i, idx in enumerate(random_indices):
    image = x_test[idx]
    actual = y_test[idx]
    
    # Format for TFLite (Batch, Width, Height)
    input_data = np.expand_dims(image, axis=0)
    
    # Measure Live Inference
    start_time = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.perf_counter()
    
    prediction = np.argmax(output[0])
    latency_ms = (end_time - start_time) * 1000
    
    # Plotting
    plt.subplot(1, 5, i+1)
    plt.imshow(image, cmap='gray')
    
    # Green text if correct, Red if wrong
    color = 'green' if prediction == actual else 'red'
    plt.title(f"Pred: {prediction}\n{latency_ms:.2f} ms", color=color, fontsize=12)
    plt.axis('off')

plt.tight_layout()
print("Demo window opened! Check your screen.")
plt.show()

print("--- Demo Completed Successfully! ---")