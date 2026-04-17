import tensorflow as tf
import numpy as np
import time
import psutil
import os

print("--- Initializing Edge AI Stress Test ---")

# 1. Setup Model
model_path = "quantized_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Setup Simulated Continuous Data Stream
num_frames = 10000
input_shape = input_details[0]['shape']
print(f"Generating {num_frames} synthetic sensor frames of shape {input_shape}...")
dummy_stream = np.random.rand(num_frames, *input_shape[1:]).astype(np.float32)

# 3. Metrics Tracking
latencies = []
process = psutil.Process(os.getpid())

# Force garbage collection to get a clean memory reading
import gc
gc.collect()
memory_before = process.memory_info().rss / (1024 * 1024) # in MB

print("\nStarting Real-Time Inference Stream (Testing 10,000 frames)...")
start_total_time = time.perf_counter()

# 4. The Real-Time Loop
for frame in dummy_stream:
    # Add batch dimension
    input_data = np.expand_dims(frame, axis=0)
    
    # Track individual frame latency
    t_start = time.perf_counter()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
    
    t_end = time.perf_counter()
    latencies.append((t_end - t_start) * 1000) # Convert to ms

end_total_time = time.perf_counter()
memory_after = process.memory_info().rss / (1024 * 1024)

# 5. Calculate Professional Benchmarks
total_time_sec = end_total_time - start_total_time
fps = num_frames / total_time_sec
avg_latency = np.mean(latencies)
p99_latency = np.percentile(latencies, 99) # The maximum latency for 99% of frames
memory_used = memory_after - memory_before

print("\n==================================================")
print("          QUALCOMM-LEVEL BENCHMARK REPORT         ")
print("==================================================")
print(f"Total Frames Processed: {num_frames}")
print(f"Throughput:             {fps:.2f} Frames Per Second (FPS)")
print(f"Average Latency:        {avg_latency:.4f} ms")
print(f"P99 Tail Latency:       {p99_latency:.4f} ms")
print(f"Active Memory Used:     ~{max(0.01, memory_used):.2f} MB")
print("==================================================")