import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset
print("Downloading and loading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the images to values between 0.0 and 1.0 (originally 0 to 255)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Print the shapes to understand the data size
print("\n--- Dataset Info ---")
print(f"Training data shape: {x_train.shape} (Images, Width, Height)")
print(f"Test data shape: {x_test.shape}")
print(f"Number of training labels: {len(y_train)}")

# 4. Visualize the first 3 images
print("\nOpening plot window... (Close the window to let the script finish)")
plt.figure(figsize=(6, 2))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()