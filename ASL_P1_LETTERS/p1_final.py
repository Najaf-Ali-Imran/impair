import cv2
import mediapipe as mp
import numpy as np
import joblib  # <-- New Import for Person C

# --- 1. LOAD THE TRAINED MODEL (Person C) ---
try:
    model = joblib.load('asl_model.joblib')
    print("✅ Model loaded successfully from 'asl_model.joblib'")
except FileNotFoundError:
    print("❌ ERROR: 'asl_model.joblib' not found.")
    print("Please make sure the model file is in the same folder as this script.")
    exit()
# --- End of New Code ---


# --- Block 2: Imports and Initial Setup (from Person B) ---
print("Libraries imported successfully.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("✅ MediaPipe and Webcam initialized.")

# --- Blocks 3, 4, & 5 combined: The Main Loop ---

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw the hand "stick figure"
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS)
            
            # Extract the 42 (x, y) coordinates
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x)
                hand_data.append(landmark.y)
            
            # --- 2. MAKE PREDICTION (Person C) ---
            # The model expects a 2D array, so we wrap `hand_data` in brackets
            prediction = model.predict([hand_data])
            predicted_letter = prediction[0]
            # --- End of New Code ---

            
            # --- 3. DISPLAY PREDICTION (Person C) ---
            # Get the coordinates for the top-left corner of the hand
            (x, y) = (hand_landmarks.landmark[0].x * frame.shape[1], 
                      hand_landmarks.landmark[0].y * frame.shape[0])
            
            # Put the text on the screen
            cv2.putText(
                frame, 
                predicted_letter, 
                (int(x), int(y) - 20), # Position 20px above the hand
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA
            )
            # --- End of New Code ---
            
    # Display the final frame
    cv2.imshow('ASL Translator', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("✅ App finished.")