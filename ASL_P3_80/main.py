
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'asl_model.h5')
CLASSES_PATH = os.path.join(SCRIPT_DIR, 'classes.npy')

MAX_FRAMES = 64              # Model expects exactly 64 frames
NUM_LANDMARKS = 121          # Total landmarks: Lips(40) + Nose(4) + Ears(2) + LeftHand(21) + Pose(33) + RightHand(21)
NUM_DIMENSIONS = 3           # x, y, z coordinates
INPUT_SIZE = NUM_LANDMARKS * NUM_DIMENSIONS  # 121 × 3 = 363 features

PREDICTION_INTERVAL = 15     # Predict every N frames
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence to display prediction


# Face mesh landmarks for lips (40 landmarks)
LIPS_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 
                314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 
                178, 87, 14, 317, 402, 318, 324, 308]

# Nose landmarks (4 landmarks)
NOSE_INDICES = [1, 2, 98, 327]

# Ear landmarks (2 landmarks)
EARS_INDICES = [33, 263]

# Left hand landmarks: indices 0-20 (21 landmarks)
LEFT_HAND_INDICES = list(range(21))

# Pose landmarks: indices 0-32 (33 landmarks for shoulders/arms)
POSE_INDICES = list(range(33))

# Right hand landmarks: indices 0-20 (21 landmarks)
RIGHT_HAND_INDICES = list(range(21))

# Total: 40 + 4 + 2 + 21 + 33 + 21 = 121 landmarks

# ================= LOAD MODEL =================
print("=" * 60)
print("ASL Real-Time Sign Language Recognition (P3_80)")
print("=" * 60)
print(f"Model Path: {MODEL_PATH}")
print(f"Classes Path: {CLASSES_PATH}")
print("Loading AI Model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    print(f"✅ Model Loaded Successfully!")
    print(f"   - Input Shape: {model.input_shape}")
    print(f"   - Number of Classes: {len(classes)}")
    print(f"   - Classes: {list(classes)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure 'asl_model.h5' and 'classes.npy' are in the same directory as this script.")
    exit(1)

# ================= MEDIAPIPE SETUP =================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_landmarks(results):
   
    landmarks = []
    
    # 1. FACE LANDMARKS (Lips + Nose + Ears = 46 landmarks)
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        # Lips (40 landmarks)
        for idx in LIPS_INDICES:
            landmarks.append([face[idx].x, face[idx].y, face[idx].z])
        # Nose (4 landmarks)
        for idx in NOSE_INDICES:
            landmarks.append([face[idx].x, face[idx].y, face[idx].z])
        # Ears (2 landmarks)
        for idx in EARS_INDICES:
            landmarks.append([face[idx].x, face[idx].y, face[idx].z])
    else:
        # Fill with zeros if face not detected (46 landmarks × 3)
        landmarks.extend([[0.0, 0.0, 0.0]] * (len(LIPS_INDICES) + len(NOSE_INDICES) + len(EARS_INDICES)))
    
    # 2. LEFT HAND (21 landmarks)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    else:
        # Fill with zeros if left hand not detected
        landmarks.extend([[0.0, 0.0, 0.0]] * 21)
    
    # 3. POSE (33 landmarks - full body pose for shoulders/arms)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    else:
        # Fill with zeros if pose not detected
        landmarks.extend([[0.0, 0.0, 0.0]] * 33)
    
    # 4. RIGHT HAND (21 landmarks)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    else:
        # Fill with zeros if right hand not detected
        landmarks.extend([[0.0, 0.0, 0.0]] * 21)
    
    return np.array(landmarks, dtype=np.float32)  # Shape: (121, 3)


def preprocess_sequence(sequence):
   
    data = np.array(sequence, dtype=np.float32)
    
    # Resize to 64 frames if needed
    if len(data) != MAX_FRAMES:
        # Reshape for tf.image.resize: (N, 121, 3) -> needs to be (N, H, W) or use resize
        data = tf.image.resize(data, (MAX_FRAMES, NUM_LANDMARKS), method='bilinear')
        data = data.numpy()
    
    # Flatten the landmarks: (64, 121, 3) -> (64, 363)
    data = data.reshape(MAX_FRAMES, INPUT_SIZE)
    
    # Normalize: subtract 0.5
    data = data - 0.5
    
    # Add batch dimension: (1, 64, 363)
    return np.expand_dims(data, axis=0)


def draw_progress_bar(frame, x, y, width, height, progress, color_bg, color_fill, label=""):
    """Draw a progress bar on the frame."""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), color_bg, -1)
    # Fill
    fill_width = int(width * min(progress, 1.0))
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color_fill, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    # Label
    if label:
        cv2.putText(frame, label, (x + width + 10, y + height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_confidence_bar(frame, x, y, width, height, confidence, label):
    # Determine color based on confidence
    if confidence >= 0.8:
        color = (0, 255, 0)      # Green - High confidence
    elif confidence >= 0.5:
        color = (0, 255, 255)    # Yellow - Medium confidence
    else:
        color = (0, 0, 255)      # Red - Low confidence
    
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    # Fill
    fill_width = int(width * confidence)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    # Label
    cv2.putText(frame, f"{label}: {confidence * 100:.1f}%", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_ui(frame, sequence, hands_detected, fps, prediction_word, confidence, top_5_predictions):
    
    h, w = frame.shape[:2]
    
    # ===== LEFT PANEL: Info & Status =====
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 280), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Title
    cv2.putText(frame, "ASL Sign Recognition (P3_80)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (350, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Recording status
    buffer_len = len(sequence)
    if buffer_len < MAX_FRAMES:
        status_text = f"RECORDING... ({buffer_len}/{MAX_FRAMES})"
        status_color = (0, 165, 255)  # Orange
    else:
        status_text = "READY - Predicting"
        status_color = (0, 255, 0)    # Green
    
    cv2.putText(frame, status_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Frame buffer progress bar (Recording indicator)
    progress = buffer_len / MAX_FRAMES
    draw_progress_bar(frame, 20, 85, 380, 20, progress, (50, 50, 50), (0, 255, 0))
    cv2.putText(frame, f"Frames: {buffer_len}/{MAX_FRAMES}", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Hands detection status
    hand_status = "DETECTED" if hands_detected else "NOT DETECTED"
    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    cv2.putText(frame, f"Hands: {hand_status}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
    
    # Prediction result
    cv2.putText(frame, "Predicted Sign:", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    if prediction_word:
        # Large prediction text
        cv2.putText(frame, prediction_word.upper(), (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Confidence bar
        draw_confidence_bar(frame, 20, 235, 380, 20, confidence, "Confidence")
    else:
        cv2.putText(frame, "Waiting for data...", (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    # ===== RIGHT PANEL: Top 5 Predictions =====
    if top_5_predictions:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w - 300, 10), (w - 10, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, "Top 5 Predictions:", (w - 290, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, (word, prob) in enumerate(top_5_predictions):
            y_pos = 65 + i * 28
            
            # Truncate long words
            display_word = word[:12] + "..." if len(word) > 12 else word
            
            # Color based on rank
            if i == 0:
                color = (0, 255, 255)  # Yellow for top prediction
            else:
                color = (200, 200, 200)
            
            # Draw mini progress bar for probability
            bar_width = int(180 * prob)
            cv2.rectangle(frame, (w - 290, y_pos - 10), (w - 110 + bar_width, y_pos + 5), 
                         (50, 100, 50), -1)
            
            cv2.putText(frame, f"{i + 1}. {display_word}", (w - 290, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(frame, f"{prob * 100:.1f}%", (w - 60, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    # ===== BOTTOM PANEL: Controls =====
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "Controls: [R] Reset Buffer | [Q] Quit | Model: 1D CNN (64 frames × 363 features)",
                (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # ===== TOP CENTER: Large Prediction Display =====
    if prediction_word and confidence > CONFIDENCE_THRESHOLD:
        # Draw large prediction at top center
        text = prediction_word.upper()
        font_scale = 1.5
        thickness = 3
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        text_x = (w - text_width) // 2
        text_y = 80
        
        # Background for text
        padding = 15
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding),
                     (0, 0, 0), -1)
        cv2.rectangle(frame,
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding),
                     (0, 255, 255), 2)
        
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    
    return frame


def draw_landmarks_on_frame(image, results):
    """Draw MediaPipe landmarks on the frame with skeleton style."""
    
    if results.face_landmarks:
        # Draw face mesh connections (simplified) - use face_mesh module for FACEMESH_LIPS
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp.solutions.face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # Draw left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    # Draw right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    return image


# ================= MAIN LOOP =================
def main():
    print("\n" + "=" * 60)
    print("Starting Webcam...")
    print("=" * 60)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Window setup
    WINDOW_NAME = 'ASL Sign Recognition (P3_80)'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    
    # Rolling buffer for frames (stores last MAX_FRAMES landmarks)
    sequence = deque(maxlen=MAX_FRAMES)
    frame_count = 0
    
    # Prediction state
    prediction_word = None
    confidence = 0.0
    top_5_predictions = []
    
    # FPS tracking
    prev_time = time.time()
    fps = 0
    
    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  R     - Reset frame buffer")
    print("  Q     - Quit")
    print("=" * 60)
    print("\n✅ Camera Ready!")
    print(f"📊 Collecting {MAX_FRAMES} frames before first prediction...")
    print("🤟 Perform signs with your hands visible!")
    
    # MediaPipe Holistic model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Process with MediaPipe Holistic
            results = holistic.process(image_rgb)
            
            # Convert back to BGR for OpenCV
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks on frame (skeleton visualization)
            image = draw_landmarks_on_frame(image, results)
            
            # Extract landmarks and add to rolling buffer
            landmarks = extract_landmarks(results)
            sequence.append(landmarks)
            frame_count += 1
            
            # Check if hands are detected
            has_hands = results.left_hand_landmarks or results.right_hand_landmarks
            
            # Run prediction every PREDICTION_INTERVAL frames when buffer is full
            if len(sequence) == MAX_FRAMES and frame_count % PREDICTION_INTERVAL == 0:
                if has_hands:
                    try:
                        # Preprocess sequence
                        input_data = preprocess_sequence(list(sequence))
                        
                        # Run inference
                        prediction = model.predict(input_data, verbose=0)
                        
                        # Get top 5 predictions
                        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
                        top_5_probs = prediction[0][top_5_indices]
                        
                        # Get best prediction
                        predicted_index = top_5_indices[0]
                        confidence = float(top_5_probs[0])
                        
                        # Build top 5 list
                        top_5_predictions = [
                            (classes[idx], float(prob)) 
                            for idx, prob in zip(top_5_indices, top_5_probs)
                        ]
                        
                        # Update prediction if confidence is high enough
                        if confidence > CONFIDENCE_THRESHOLD:
                            prediction_word = classes[predicted_index]
                            print(f"🎯 Detected: {prediction_word.upper()} ({confidence * 100:.1f}%)")
                        else:
                            prediction_word = f"({classes[predicted_index]}?)"  # Low confidence indicator
                            
                    except Exception as e:
                        print(f"⚠️ Prediction error: {e}")
                else:
                    # No hands detected - clear prediction
                    prediction_word = None
                    top_5_predictions = []
            
            # Draw UI overlay
            image = draw_ui(image, sequence, has_hands, fps, prediction_word, confidence, top_5_predictions)
            
            # Display frame
            cv2.imshow(WINDOW_NAME, image)
            
            # Handle keyboard input
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('r') or key == ord('R'):
                # Reset buffer
                sequence.clear()
                prediction_word = None
                confidence = 0.0
                top_5_predictions = []
                frame_count = 0
                print("\n🔄 Buffer reset - Start recording new gesture")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
