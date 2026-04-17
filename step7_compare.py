import tensorflow as tf
import numpy as np
import os

print("--- Starting Final Comparison ---")

# 1. Load test data (we'll use 1,000 images to keep the test fast)
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = (x_test[:1000] / 255.0).astype(np.float32)
y_test = y_test[:1000]

# 2. Function to evaluate a TFLite model
def evaluate_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct_predictions = 0
    
    # Loop through our 1,000 test images
    for i in range(len(x_test)):
        interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output[0]) # Get the highest probability digit
        
        if prediction == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(x_test)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return size_mb, accuracy

# 3. Evaluate both models
print("Testing Unoptimized TFLite Model... (Wait a moment)")
tflite_size, tflite_acc = evaluate_tflite_model("unoptimized_model.tflite")

print("Testing Quantized TFLite Model... (Wait a moment)")
quant_size, quant_acc = evaluate_tflite_model("quantized_model.tflite")

# 4. Print the Comparison Table
print("\n=====================================================")
print("               OPTIMIZATION RESULTS                  ")
print("=====================================================")
print(f"Metric           | Unoptimized      | Quantized ")
print("-----------------------------------------------------")
print(f"File Size        | {tflite_size:.2f} MB          | {quant_size:.2f} MB")
print(f"Accuracy         | {tflite_acc * 100:.2f} %          | {quant_acc * 100:.2f} %")
print("-----------------------------------------------------")
print("Note: Quantization achieved this size reduction by")
print("converting 32-bit floats to 8-bit integers!")
print("=====================================================")