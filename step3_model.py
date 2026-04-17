import tensorflow as tf
import time
import os

# 1. Load and normalize data (same as before)
print("Loading data...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build the Neural Network
# We flatten the 28x28 image into a 1D array, pass it to a hidden layer of 128 neurons, 
# and output to 10 neurons (representing digits 0-9)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
print("\n--- Training Base Model ---")
model.fit(x_train, y_train, epochs=3)

# 5. Evaluate Accuracy
print("\n--- Model Evaluation ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Base Accuracy: {accuracy * 100:.2f}%")

# 6. Measure Inference Time
print("\n--- Measuring Inference Time ---")
# "Warm up" the model (TensorFlow is often slow on the very first prediction)
_ = model.predict(x_test[:1], verbose=0)

# Time how long it takes to predict 1,000 images
start_time = time.time()
_ = model.predict(x_test[:1000], verbose=0)
end_time = time.time()

total_time_sec = end_time - start_time
time_per_image_ms = (total_time_sec / 1000) * 1000  # Convert to milliseconds

print(f"Average Inference Time per image: {time_per_image_ms:.4f} milliseconds")

# 7. Show Model Architecture (Parameters dictate size)
print("\n--- Model Architecture ---")
model.summary()



# 8. Save the model to disk
print("\n--- Saving Model ---")
model_path = "base_model.keras"
model.save(model_path)
print(f"Model saved as '{model_path}'")

# 9. Measure the actual file size
size_bytes = os.path.getsize(model_path)
size_mb = size_bytes / (1024 * 1024)

# 10. Print the Final Baseline Benchmark
print("\n==================================")
print("      BASELINE BENCHMARK         ")
print("==================================")
print(f"Model Size:        {size_mb:.2f} MB")
print(f"Accuracy:          {accuracy * 100:.2f} %")
print(f"Inference Time:    {time_per_image_ms:.4f} ms / image")
print("==================================")