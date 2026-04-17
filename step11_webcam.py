import cv2
import numpy as np
import tensorflow as tf
import time

print("--- Launching Live Edge Camera Feed ---")
print("Press 'q' in the camera window to quit.")

# 1. Load the Quantized Model
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Open the Webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)

# Variables for FPS tracking
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)
    
    # 3. Define a Region of Interest (ROI) - a box in the center
    height, width, _ = frame.shape
    # Draw a 300x300 box
    top_left_x = width // 2 - 150
    top_left_y = height // 2 - 150
    bottom_right_x = width // 2 + 150
    bottom_right_y = height // 2 + 150
    
    # Draw the box on the main frame (Color: Green, Thickness: 2)
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    
    # 4. Extract and Preprocess the ROI
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Blur slightly to remove camera noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Thresholding: MNIST expects white digits on a black background!
    # THRESH_BINARY_INV inverts the colors so black ink on white paper becomes white on black.
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Resize to the 28x28 size our model expects
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize to 0.0 - 1.0 (float32)
    input_data = (resized / 255.0).astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0) # Add batch dimension
    
    # 5. Run Live Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output[0])
    confidence = output[0][prediction] * 100
    
    # 6. Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # 7. Display Results on the Screen
    cv2.putText(frame, f"Prediction: {prediction} ({confidence:.0f}%)", (top_left_x, top_left_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Edge FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the main camera view
    cv2.imshow("Edge AI Live Feed", frame)
    
    # Show what the model actually sees (the tiny 28x28 inverted box, scaled up so you can see it)
    cv2.imshow("Model View (Preprocessed)", cv2.resize(thresh, (150, 150)))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()