"""
ASL Real-Time Continuous Recognition
=====================================
Continuously recognize ASL signs using webcam with MediaPipe landmark extraction.

Controls:
- Press 'R' to reset buffer
- Press 'Q' to quit..
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time

# ================= CONFIG =================
MODEL_PATH = 'asl_model.h5'
CLASSES_PATH = 'classes.npy'
MAX_FRAMES = 64  # Input size for the model
NUM_LANDMARKS = 115 
INPUT_SIZE = NUM_LANDMARKS * 2 
PREDICTION_INTERVAL = 15 # Predict every 15 frames (0.5s) to avoid lag
CONFIDENCE_THRESHOLD = 0.75 # Only show if very sure

# ================= LOAD MODEL =================
print("="*60)
print("ASL Real-Time Continuous Recognition")
print("="*60)
print("Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)
print(f"✅ Model Loaded. Recognizing {len(classes)} signs.")

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ================= HELPER FUNCTIONS =================
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LH = list(range(468, 489))
POSE = list(range(489, 522))
RH = list(range(522, 543))
KEEP_INDICES = LIPS + LH + POSE + RH

def extract_landmarks(results):
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 2))
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 2))
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 2))
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 2))
    
    all_landmarks = np.concatenate([face, lh, pose, rh])
    return all_landmarks[KEEP_INDICES]

def preprocess_sequence(sequence):
    data = np.array(sequence)
    # Resize to 64 frames (Input shape: (N, 115, 2) -> (64, 115, 2))
    data = tf.image.resize(data, (MAX_FRAMES, NUM_LANDMARKS), method='bilinear')
    # Flatten (64, 230)
    data = tf.reshape(data, (MAX_FRAMES, INPUT_SIZE))
    data = data - 0.5 
    return np.expand_dims(data, axis=0)

def draw_ui(frame, sequence, hands_detected, fps, prediction_word, confidence, top_5_predictions):
    """Draw UI elements on frame similar to test_webcam.py style."""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay for info panel (left side)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Title
    cv2.putText(frame, "ASL Sign Recognition", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (320, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Recording status (always recording in continuous mode)
    status_color = (0, 255, 0)
    status_text = "CONTINUOUS MODE"
    cv2.putText(frame, status_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Frame buffer progress bar
    buffer_len = len(sequence)
    progress = buffer_len / MAX_FRAMES
    cv2.rectangle(frame, (20, 80), (380, 95), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 80), (20 + int(360 * progress), 95), (0, 255, 0), -1)
    cv2.putText(frame, f"Frames: {buffer_len}/{MAX_FRAMES}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Hands detection status
    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    hand_text = "Hands: DETECTED" if hands_detected else "Hands: NOT DETECTED"
    cv2.putText(frame, hand_text, (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
    
    # Prediction result
    if prediction_word:
        cv2.putText(frame, f"Sign: {prediction_word.upper()}", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Sign: Waiting...", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    # Top 5 predictions panel (right side)
    if top_5_predictions:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w-280, 10), (w-10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "Top 5 Predictions:", (w-270, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (word, prob) in enumerate(top_5_predictions):
            y_pos = 60 + i * 25
            # Truncate long words for display
            display_word = word[:15] + "..." if len(word) > 15 else word
            cv2.putText(frame, f"{i+1}. {display_word}: {prob*100:.1f}%", (w-270, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Controls help (bottom)
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (10, h-60), (w-10, h-10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay3, 0.6, frame, 0.4, 0)
    
    cv2.putText(frame, "Controls: [R] Reset Buffer | [Q] Quit | Continuous prediction when hands detected", 
                (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

# ================= MAIN LOOP =================
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Window Setup
WINDOW_NAME = 'ASL Continuous Recognition'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# Rolling Buffer (Stores last 64 frames)
sequence = deque(maxlen=MAX_FRAMES)
frame_count = 0

# Prediction state
prediction_word = None
confidence = 0.0
top_5_predictions = []

# FPS tracking
prev_time = time.time()
fps = 0

print("\n" + "="*60)
print("CONTROLS:")
print("  R     - Reset frame buffer")
print("  Q     - Quit")
print("="*60)
print("\n✅ Starting Camera... Perform signs continuously.")
print("Make sure your hands are visible!")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("Failed to read frame")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # 1. Process Frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. Draw Landmarks with better styling
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # 3. Add to Rolling Buffer
        landmarks = extract_landmarks(results)
        sequence.append(landmarks)
        frame_count += 1

        # Check for hands
        has_hands = results.left_hand_landmarks or results.right_hand_landmarks

        # 4. Predict Every X Frames (Only if buffer is full)
        if len(sequence) == MAX_FRAMES and frame_count % PREDICTION_INTERVAL == 0:
            if has_hands:
                input_data = preprocess_sequence(list(sequence))
                prediction = model.predict(input_data, verbose=0)
                
                # Get top 5 predictions
                top_5_indices = np.argsort(prediction[0])[-5:][::-1]
                top_5_probs = prediction[0][top_5_indices]
                
                predicted_index = top_5_indices[0]
                confidence = top_5_probs[0]
                
                # Build top 5 list
                top_5_predictions = [(classes[idx], prob) for idx, prob in zip(top_5_indices, top_5_probs)]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    prediction_word = classes[predicted_index]
                    print(f"🎯 Detected: {prediction_word.upper()} ({confidence*100:.1f}%)")
                else:
                    prediction_word = None
            else:
                prediction_word = None
                top_5_predictions = []

        # 5. Draw UI
        image = draw_ui(image, sequence, has_hands, fps, prediction_word, confidence, top_5_predictions)

        cv2.imshow(WINDOW_NAME, image)

        # Handle key presses
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            sequence.clear()
            prediction_word = None
            confidence = 0.0
            top_5_predictions = []
            print("\n🔄 Buffer reset")

cap.release()
cv2.destroyAllWindows()
print("✅ Cleanup complete")