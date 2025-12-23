"""
ASL Detection Engine - Uses existing optimized implementations
Integrates Phase 1 (Letters), Phase 2 (5 Words), and Phase 3 (80 Words)
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
import os
import time

class DetectionEngine(QThread):
    # Signals
    frame_ready = pyqtSignal(QImage)
    prediction_ready = pyqtSignal(str, float)  # text, confidence
    hand_detected = pyqtSignal(bool)  # hand detection status
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.current_mode = "words5"  # Default to Phrase mode
        self.debug_mode = False
        
        # Camera
        self.cap = None
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = None
        self.holistic = None
        
        # Frame tracking
        self.frame_count = 0
        
        # Load all models
        self.load_models()
    
    def load_models(self):
        """Load all three phase models"""
        try:
            # Phase 1 - Letters
            p1_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'ASL_P1_LETTERS', 'asl_model.joblib')
            self.model_letters = joblib.load(p1_path)
            self.prediction_interval_p1 = 5
            print("✅ Phase 1 (Letters) model loaded")
            
            # Phase 2 - 5 Words
            p2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'ASL_P2_5', 'my_lstm_model_phase2.h5')
            self.model_words5 = tf.keras.models.load_model(p2_path)
            self.actions_p2 = np.array(['goodbye', 'hello', 'me', 'thanks', 'you'])
            self.sequence_p2 = deque(maxlen=40)
            self.prediction_interval_p2 = 10
            self.threshold_p2 = 0.8
            print("✅ Phase 2 (5 Words) model loaded")
            
            # Phase 3 - 80 Words
            p3_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                        'ASL_P3_80', 'asl_model.h5')
            p3_classes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                          'ASL_P3_80', 'classes.npy')
            self.model_words80 = tf.keras.models.load_model(p3_model_path)
            self.classes_p3 = np.load(p3_classes_path, allow_pickle=True)
            self.sequence_p3 = deque(maxlen=64)
            self.prediction_interval_p3 = 15
            self.threshold_p3 = 0.5
            
            # P3 landmark indices
            self.lips_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 
                                314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 
                                178, 87, 14, 317, 402, 318, 324, 308]
            self.nose_indices = [1, 2, 98, 327]
            self.ears_indices = [33, 263]
            
            # P3 constants from ASL_P3_80/main.py
            self.max_frames_p3 = 64
            self.num_landmarks_p3 = 121
            self.num_dimensions_p3 = 3
            self.input_size_p3 = self.num_landmarks_p3 * self.num_dimensions_p3  # 363
            
            print(f"✅ Phase 3 (80 Words) model loaded - {len(self.classes_p3)} classes")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
    
    def set_mode(self, mode):
        """Switch between detection modes"""
        was_running = self.running
        
        # Close existing MediaPipe instances
        if self.hands:
            self.hands.close()
            self.hands = None
        if self.holistic:
            self.holistic.close()
            self.holistic = None
        
        self.current_mode = mode
        self.frame_count = 0
        if mode == "words5":
            self.sequence_p2.clear()
        elif mode == "words80":
            self.sequence_p3.clear()
        
        # Reinitialize MediaPipe if detection was running
        if was_running:
            if mode in ["letters", "words5"]:
                max_hands = 1 if mode == "letters" else 2
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print(f"✅ MediaPipe Hands reinitialized for {mode}")
            else:  # words80
                self.holistic = self.mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1,
                    refine_face_landmarks=True  # CRITICAL: Enable full face mesh for lips/nose/ears
                )
                print(f"✅ MediaPipe Holistic reinitialized for {mode}")
        
        print(f"🔄 Switched to mode: {mode}")
    
    def set_debug_mode(self, enabled):
        """Toggle debug visualization"""
        self.debug_mode = enabled
    
    def start_detection(self):
        """Start camera and detection"""
        try:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("❌ Error: Could not open camera")
                    return
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Close any existing MediaPipe instances
            if self.hands:
                self.hands.close()
                self.hands = None
            if self.holistic:
                self.holistic.close()
                self.holistic = None
            
            # Initialize MediaPipe based on mode
            if self.current_mode in ["letters", "words5"]:
                max_hands = 1 if self.current_mode == "letters" else 2
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print(f"✅ MediaPipe Hands initialized for {self.current_mode}")
            else:  # words80
                self.holistic = self.mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_complexity=1,
                    refine_face_landmarks=True  # CRITICAL: Enable full face mesh for lips/nose/ears
                )
                print(f"✅ MediaPipe Holistic initialized for {self.current_mode}")
            
            self.running = True
            self.start()
            print("✅ Detection started")
        except Exception as e:
            print(f"❌ Error starting detection: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_detection(self):
        """Stop camera and detection"""
        self.running = False
        self.wait()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.hands:
            self.hands.close()
            self.hands = None
        if self.holistic:
            self.holistic.close()
            self.holistic = None
        print("⏹️ Detection stopped")
    
    def run(self):
        """Main detection loop"""
        try:
            while self.running:
                if self.cap is None or not self.cap.isOpened():
                    print("❌ Camera not available")
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process based on current mode
                if self.current_mode == "letters":
                    self.process_p1_letters(frame, rgb_frame)
                elif self.current_mode == "words5":
                    self.process_p2_words5(frame, rgb_frame)
                elif self.current_mode == "words80":
                    self.process_p3_words80(frame, rgb_frame)
                
                self.frame_count += 1
        except Exception as e:
            print(f"❌ Error in detection loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()
    
    # ==================== PHASE 1: LETTERS ====================
    def process_p1_letters(self, frame, rgb_frame):
        """Phase 1 - Letter Recognition"""
        if self.hands is None or self.model_letters is None:
            return
        
        results = self.hands.process(rgb_frame)
        hands_detected = results.multi_hand_landmarks is not None
        self.hand_detected.emit(hands_detected)
        
        # Draw landmarks if debug mode
        if self.debug_mode and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Predict every N frames
        if results.multi_hand_landmarks and self.frame_count % self.prediction_interval_p1 == 0:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x)
                hand_data.append(landmark.y)
            
            prediction = self.model_letters.predict([hand_data])
            predicted_letter = prediction[0].upper()
            confidence = 0.95  # joblib doesn't return confidence
            
            self.prediction_ready.emit(predicted_letter, confidence)
        
        self.emit_frame(frame)
    
    # ==================== PHASE 2: 5 WORDS ====================
    def extract_keypoints_p2(self, results):
        """Extract 126 features for Phase 2"""
        lh = np.zeros(63)
        rh = np.zeros(63)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                points = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                if handedness.classification[0].label == 'Left':
                    lh = points
                else:
                    rh = points
        return np.concatenate([rh, lh])
    
    def process_p2_words5(self, frame, rgb_frame):
        """Phase 2 - 5 Words Recognition"""
        if self.hands is None or self.model_words5 is None:
            return
        
        results = self.hands.process(rgb_frame)
        hands_detected = results.multi_hand_landmarks is not None
        self.hand_detected.emit(hands_detected)
        
        # Draw landmarks if debug mode
        if self.debug_mode and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Extract keypoints and add to sequence
        keypoints = self.extract_keypoints_p2(results)
        self.sequence_p2.append(keypoints)
        
        # Predict when buffer is full
        if len(self.sequence_p2) == 40 and self.frame_count % self.prediction_interval_p2 == 0:
            if hands_detected:
                res = self.model_words5.predict(np.expand_dims(list(self.sequence_p2), axis=0), verbose=0)[0]
                predicted_index = np.argmax(res)
                confidence = float(res[predicted_index])
                
                if confidence > self.threshold_p2:
                    predicted_word = self.actions_p2[predicted_index].capitalize()
                    self.prediction_ready.emit(predicted_word, confidence)
        
        self.emit_frame(frame)
    
    # ==================== PHASE 3: 80 WORDS ====================
    def extract_keypoints_p3(self, results):
        """
        Extract exactly 121 landmarks × 3 dimensions = 363 features.
        EXACT COPY from ASL_P3_80/main.py
        
        Order (matching training data):
        1. Lips (40 landmarks from face mesh)
        2. Nose (4 landmarks from face mesh)
        3. Ears (2 landmarks from face mesh)
        4. Left Hand (21 landmarks)
        5. Pose (33 landmarks - shoulders/arms)
        6. Right Hand (21 landmarks)
        
        Returns: numpy array of shape (121, 3)
        """
        landmarks = []
        
        # 1. FACE LANDMARKS (Lips + Nose + Ears = 46 landmarks)
        if results.face_landmarks:
            face = results.face_landmarks.landmark
            # Lips (40 landmarks)
            for idx in self.lips_indices:
                landmarks.append([face[idx].x, face[idx].y, face[idx].z])
            # Nose (4 landmarks)
            for idx in self.nose_indices:
                landmarks.append([face[idx].x, face[idx].y, face[idx].z])
            # Ears (2 landmarks)
            for idx in self.ears_indices:
                landmarks.append([face[idx].x, face[idx].y, face[idx].z])
        else:
            # Fill with zeros if face not detected (46 landmarks × 3)
            landmarks.extend([[0.0, 0.0, 0.0]] * (len(self.lips_indices) + len(self.nose_indices) + len(self.ears_indices)))
        
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
    
    def preprocess_sequence_p3(self, sequence):
        """
        Preprocess the sequence of landmarks for model input.
        EXACT COPY from ASL_P3_80/main.py
        
        Steps:
        1. Stack sequence into array
        2. Resize to exactly 64 frames (if needed)
        3. Flatten landmarks: (64, 121, 3) -> (64, 363)
        4. Normalize by subtracting 0.5
        5. Add batch dimension: (1, 64, 363)
        
        Args:
            sequence: List of numpy arrays, each of shape (121, 3)
        
        Returns:
            numpy array of shape (1, 64, 363)
        """
        # Stack into array: (N, 121, 3)
        data = np.array(sequence, dtype=np.float32)
        
        # Resize to 64 frames if needed
        if len(data) != self.max_frames_p3:
            data = tf.image.resize(data, (self.max_frames_p3, self.num_landmarks_p3), method='bilinear')
            data = data.numpy()
        
        # Flatten the landmarks: (64, 121, 3) -> (64, 363)
        data = data.reshape(self.max_frames_p3, self.input_size_p3)
        
        # Normalize: subtract 0.5
        data = data - 0.5
        
        # Add batch dimension: (1, 64, 363)
        return np.expand_dims(data, axis=0)
    
    def process_p3_words80(self, frame, rgb_frame):
        """Phase 3 - 80 Words Recognition"""
        if self.holistic is None:
            print("⚠️ Holistic not initialized")
            return
        if self.model_words80 is None:
            print("⚠️ Model not loaded")
            return
        
        try:
            # CRITICAL: Set writeable flag to False for MediaPipe optimization (from ASL_P3_80/main.py)
            rgb_frame.flags.writeable = False
            results = self.holistic.process(rgb_frame)
            # Set back to True after processing
            rgb_frame.flags.writeable = True
        except Exception as e:
            print(f"❌ Error processing holistic: {e}")
            self.emit_frame(frame)
            return
        
        hands_detected = (results.left_hand_landmarks is not None or 
                         results.right_hand_landmarks is not None)
        self.hand_detected.emit(hands_detected)
        
        # Draw landmarks if debug mode
        if self.debug_mode:
            if results.face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style()
                )
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style()
                )
        
        # Extract keypoints and add to sequence
        keypoints = self.extract_keypoints_p3(results)  # Shape: (121, 3)
        self.sequence_p3.append(keypoints)
        
        # Predict when buffer is full
        if len(self.sequence_p3) == 64 and self.frame_count % self.prediction_interval_p3 == 0:
            if hands_detected:
                try:
                    # Preprocess sequence - EXACT SAME AS ASL_P3_80/main.py
                    input_data = self.preprocess_sequence_p3(list(self.sequence_p3))
                    
                    # Run prediction
                    res = self.model_words80.predict(input_data, verbose=0)[0]
                    predicted_index = np.argmax(res)
                    confidence = float(res[predicted_index])
                    
                    if confidence > self.threshold_p3:
                        predicted_word = self.classes_p3[predicted_index].capitalize()
                        self.prediction_ready.emit(predicted_word, confidence)
                except Exception as e:
                    print(f"⚠️ P3 Prediction error: {e}")
                    import traceback
                    traceback.print_exc()
        
        self.emit_frame(frame)
    
    def emit_frame(self, frame):
        """Convert OpenCV frame to QImage and emit"""
        try:
            if frame is None or frame.size == 0:
                return
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qt_image)
        except Exception as e:
            print(f"❌ Error emitting frame: {e}")
