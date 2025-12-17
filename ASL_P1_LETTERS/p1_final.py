"""
ASL Real-Time Letter Recognition - Phase 1
===========================================
Recognize ASL letters using webcam with MediaPipe hand landmarks.

Controls:
- Press 'R' to reset/clear
- Press 'Q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# --- CONFIGURATION ---
MODEL_PATH = 'asl_model.joblib'
PREDICTION_INTERVAL = 5  # Predict every 5 frames to reduce lag

# --- LOAD THE TRAINED MODEL ---
print("="*60)
print("ASL Real-Time Letter Recognition - Phase 1")
print("="*60)
print("Loading AI Model...")
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"❌ ERROR: '{MODEL_PATH}' not found.")
    print("Please make sure the model file is in the same folder as this script.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize OpenCV Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Window Setup
WINDOW_NAME = 'ASL Phase 1 - Letter Recognition'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# Variables
frame_count = 0
prediction_letter = None
confidence = 0.0
prediction_history = []

# FPS tracking
prev_time = time.time()
fps = 0

print("\n✅ MediaPipe and Webcam initialized.")

# --- HELPER FUNCTION ---
def draw_ui(frame, hands_detected, fps, prediction_letter, prediction_history):
    """Draw UI elements on frame similar to other ASL projects."""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay for info panel (left side)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Title
    cv2.putText(frame, "ASL Letter Recognition", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (320, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Recording status
    status_color = (0, 255, 0)
    status_text = "REAL-TIME MODE"
    cv2.putText(frame, status_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Hands detection status
    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    hand_text = "Hand: DETECTED" if hands_detected else "Hand: NOT DETECTED"
    cv2.putText(frame, hand_text, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
    
    # Prediction result
    if prediction_letter:
        cv2.putText(frame, f"Letter: {prediction_letter.upper()}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Letter: Waiting...", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Prediction history (right side)
    if prediction_history:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w-280, 10), (w-10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "Recent Letters:", (w-270, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show last 5 predictions
        display_history = prediction_history[-5:]
        for i, letter in enumerate(display_history):
            y_pos = 60 + i * 25
            cv2.putText(frame, f"{len(prediction_history)-len(display_history)+i+1}. {letter.upper()}", 
                        (w-270, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Controls help (bottom)
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (10, h-60), (w-10, h-10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay3, 0.6, frame, 0.4, 0)
    
    cv2.putText(frame, "Controls: [R] Clear History | [Q] Quit | Real-time letter recognition", 
                (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

print("\n" + "="*60)
print("CONTROLS:")
print("  R     - Clear prediction history")
print("  Q     - Quit")
print("="*60)
print("\n✅ Starting Camera... Show hand signs for letters.")
print("Make sure your hand is visible!")

# --- MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Check for hands
    has_hands = results.multi_hand_landmarks is not None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw the hand skeleton with better styling
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Predict every N frames to reduce lag
            if frame_count % PREDICTION_INTERVAL == 0:
                # Extract the 42 (x, y) coordinates
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append(landmark.x)
                    hand_data.append(landmark.y)
                
                # Make prediction
                prediction = model.predict([hand_data])
                predicted_letter = prediction[0]
                prediction_letter = predicted_letter
                
                # Add to history if it's a new prediction
                if not prediction_history or prediction_history[-1] != predicted_letter:
                    prediction_history.append(predicted_letter)
                    print(f"🎯 Detected: {predicted_letter.upper()}")
            
    else:
        prediction_letter = None
    
    frame_count += 1
    
    # Draw UI
    image = draw_ui(image, has_hands, fps, prediction_letter, prediction_history)
    
    # Display the final frame
    cv2.imshow(WINDOW_NAME, image)

    # Handle key presses
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('r'):
        prediction_history.clear()
        prediction_letter = None
        frame_count = 0
        print("\n🔄 History cleared")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("✅ Cleanup complete")