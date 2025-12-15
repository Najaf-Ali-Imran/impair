import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# --- CONFIGURATION ---
# 1. Matches your file name exactly
MODEL_PATH = 'my_lstm_model_phase2.h5' 

# 2. Your specific classes (Alphabetical order is safest for verification)
# Try this order first. If words are swapped, we just rearrange this list.
ACTIONS = np.array(['goodbye', 'hello', 'me', 'thanks', 'you'])

# 3. Model Parameters
SEQUENCE_LENGTH = 40  # Your model needs 40 frames of history
THRESHOLD = 0.8       # Confidence required to show text

# --- LOAD RESOURCES ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except OSError:
    print(f"❌ ERROR: Could not find '{MODEL_PATH}'. Please check the folder.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Try opening camera (0 is default, use 1 if external)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Variables
sequence = [] 
sentence = []

# --- HELPER FUNCTION ---
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

print("📷 Camera starting... Press 'q' to quit.")

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not readable")
        break

    # 1. Processing
    image = cv2.flip(frame, 1) # Mirror effect
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # 2. Draw Skeleton
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 3. Prediction Logic
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:] # Keep exactly the last 40 frames

    if len(sequence) == SEQUENCE_LENGTH:
        # Ask model to predict
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        
        # Get the highest score
        best_score_index = np.argmax(res)
        confidence = res[best_score_index]

        # Debug print (Optional: remove later)
        # print(f"Prediction: {ACTIONS[best_score_index]} ({confidence:.2f})")

        if confidence > THRESHOLD:
            current_word = ACTIONS[best_score_index]
            
            # Display Word + Score
            text = f"{current_word} ({int(confidence*100)}%)"
            
            # Green Box
            cv2.rectangle(image, (0,0), (640, 50), (0, 200, 0), -1)
            # White Text
            cv2.putText(image, text, (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Show Video
    cv2.imshow('ASL Phase 2 Test', image)

    # Quit Key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()