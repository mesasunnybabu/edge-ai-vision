import tensorflow as tf
import numpy as np
import time

print("--- Booting Edge Device Simulation ---")
print("Initializing camera feed simulation...\n")

# 1. Load the ultra-lightweight Quantized model
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Get 20 "frames" to act as our live sensor data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
frames = (x_test[:20] / 255.0).astype(np.float32)
actual_labels = y_test[:20]

print("Starting Real-Time Inference Loop:\n")
total_latency = 0

# 3. Simulate processing data as it arrives in real-time
for i, frame in enumerate(frames):
    start_time = time.perf_counter()
    
    # We must expand dimensions because the model expects (Batch, Width, Height)
    # Even if Batch is just 1 image.
    input_data = np.expand_dims(frame, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output[0])
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    total_latency += latency_ms
    
    # Print the live result
    status = "CORRECT" if prediction == actual_labels[i] else "WRONG"
    print(f"Frame {i+1:02d} | Predicted: {prediction} ({status}) | Latency: {latency_ms:.3f} ms")
    
    # Simulate a 30 FPS camera delay (waiting ~33ms for the next physical frame to arrive)
    time.sleep(0.033) 

avg_latency = total_latency / len(frames)
print(f"\n--- Simulation Complete ---")
print(f"Average Live Latency: {avg_latency:.3f} ms per frame")