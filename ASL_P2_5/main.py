"""
ASL Real-Time Continuous Recognition - Phase 2
===============================================
Continuously recognize ASL signs (5 words) using webcam with MediaPipe hand landmarks.

Controls:
- Press 'R' to reset buffer
- Press 'Q' to quit
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time

# --- CONFIGURATION ---
# 1. Matches your file name exactly
MODEL_PATH = 'my_lstm_model_phase2.h5' 

# 2. Your specific classes (Alphabetical order is safest for verification)
# Try this order first. If words are swapped, we just rearrange this list.
ACTIONS = np.array(['goodbye', 'hello', 'me', 'thanks', 'you'])

# 3. Model Parameters
SEQUENCE_LENGTH = 40  # Your model needs 40 frames of history
THRESHOLD = 0.8       # Confidence required to show text
PREDICTION_INTERVAL = 10  # Predict every 10 frames to avoid lag

# --- LOAD RESOURCES ---
print("="*60)
print("ASL Real-Time Continuous Recognition - Phase 2")
print("="*60)
print("Loading AI Model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model Loaded. Recognizing {len(ACTIONS)} signs.")
except OSError:
    print(f"❌ ERROR: Could not find '{MODEL_PATH}'. Please check the folder.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Try opening camera (0 is default, use 1 if external)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Window Setup
WINDOW_NAME = 'ASL Phase 2 - Continuous Recognition'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# Variables
sequence = deque(maxlen=SEQUENCE_LENGTH)
frame_count = 0

# Prediction state
prediction_word = None
confidence = 0.0
top_5_predictions = []

# FPS tracking
prev_time = time.time()
fps = 0

# --- HELPER FUNCTIONS ---
def extract_keypoints(results):
    """
    Extracts 126 features (63 Left + 63 Right).
    If a hand is missing, it fills the data with zeros.
    """
    lh = np.zeros(63)
    rh = np.zeros(63)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Flatten 21 x,y,z points into 63 numbers
            points = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if handedness.classification[0].label == 'Left':
                lh = points
            else:
                rh = points
    # Combine (Right Hand first, then Left Hand is a common training standard)
    return np.concatenate([rh, lh]) 

def draw_ui(frame, sequence, hands_detected, fps, prediction_word, confidence, top_5_predictions):
    """Draw UI elements on frame similar to ASL_P3_30 style."""
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
    progress = buffer_len / SEQUENCE_LENGTH
    cv2.rectangle(frame, (20, 80), (380, 95), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 80), (20 + int(360 * progress), 95), (0, 255, 0), -1)
    cv2.putText(frame, f"Frames: {buffer_len}/{SEQUENCE_LENGTH}", (20, 115),
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

print("\n" + "="*60)
print("CONTROLS:")
print("  R     - Reset frame buffer")
print("  Q     - Quit")
print("="*60)
print("\n✅ Starting Camera... Perform signs continuously.")
print("Make sure your hands are visible!")

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not readable")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # 1. Processing
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 2. Draw Skeleton with better styling
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # 3. Add to Rolling Buffer
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    frame_count += 1

    # Check for hands
    has_hands = results.multi_hand_landmarks is not None

    # 4. Predict Every X Frames (Only if buffer is full)
    if len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICTION_INTERVAL == 0:
        if has_hands:
            # Ask model to predict
            res = model.predict(np.expand_dims(list(sequence), axis=0), verbose=0)[0]
            
            # Get top 5 predictions
            top_5_indices = np.argsort(res)[-5:][::-1]
            top_5_probs = res[top_5_indices]
            
            predicted_index = top_5_indices[0]
            confidence = top_5_probs[0]
            
            # Build top 5 list
            top_5_predictions = [(ACTIONS[idx], prob) for idx, prob in zip(top_5_indices, top_5_probs)]
            
            if confidence > THRESHOLD:
                prediction_word = ACTIONS[predicted_index]
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
        frame_count = 0
        print("\n🔄 Buffer reset")

cap.release()
cv2.destroyAllWindows()
print("✅ Cleanup complete")