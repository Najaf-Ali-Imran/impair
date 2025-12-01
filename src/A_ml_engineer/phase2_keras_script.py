!pip install tensorflow numpy pandas tqdm

from google.colab import drive
import os
import joblib

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

print("✅ Drive Mounted.")


import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import warnings
import zipfile
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION (FINAL, DEFINITIVE PATHS) ---
BASE_DRIVE_PATH = '/content/drive/MyDrive/ASL_Project_Files'
ZIP_FILE_NAME = 'DATASET_P2.zip'
EXTRACT_FOLDER = 'DATASET_P2_EXTRACTED_FINAL' # New clean extraction folder name

# Path to the base of the extracted content
EXTRACTED_BASE_PATH = os.path.join(BASE_DRIVE_PATH, EXTRACT_FOLDER)

# --- 1. UNZIPPING DATASET (CLEAN EXTRACTION) ---
print("--- 1. UNZIPPING DATASET ---")

try:
    # Ensure all imports are ready
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')

    os.chdir(BASE_DRIVE_PATH)

    if os.path.exists(ZIP_FILE_NAME):
        # Create the extraction folder if it doesn't exist
        os.makedirs(EXTRACTED_BASE_PATH, exist_ok=True)

        # Extract the contents into the clean new folder name
        with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_BASE_PATH)
        print("✅ ZIP extracted successfully into new clean folder.")

        # --- DIAGNOSTIC STEP: Print contents of the extracted folder ---
        print(f"--- Contents of {EXTRACTED_BASE_PATH}:")
        for item in os.listdir(EXTRACTED_BASE_PATH):
            print(f" - {item}")
        print("--------------------------------------------------")

    else:
        raise FileNotFoundError(f"❌ FATAL ERROR: ZIP file '{ZIP_FILE_NAME}' not found at {BASE_DRIVE_PATH}.")

except Exception as e:
    print(f"❌ ERROR during setup/unzip: {e}")
    # Instead of exit(), re-raise the exception to stop execution and provide a traceback
    raise

# --- 2. FEATURE EXTRACTION & CONSOLIDATION ---

print("\n--- 2. FEATURE EXTRACTION & CONSOLIDATION ---")

# --- Dynamic ACTION_CLASS_PATH determination ---
# Try to find the 'MP_Data' folder. It could be directly under EXTRACTED_BASE_PATH
# or nested under a 'DATASET_P2' folder or 'dataset P2' (with a space).

possible_action_paths = [
    os.path.join(EXTRACTED_BASE_PATH, 'dataset P2', 'MP_Data'), # Corrected path based on diagnostic output
    os.path.join(EXTRACTED_BASE_PATH, 'DATASET_P2', 'MP_Data'),
    os.path.join(EXTRACTED_BASE_PATH, 'MP_Data'),
]

ACTION_CLASS_PATH = None
for path_attempt in possible_action_paths:
    if os.path.exists(path_attempt):
        ACTION_CLASS_PATH = path_attempt
        break

if ACTION_CLASS_PATH is None:
    raise FileNotFoundError(f"❌ FATAL ERROR: 'MP_Data' structure not found under {EXTRACTED_BASE_PATH}. Checked paths: {possible_action_paths}. Please verify your ZIP file's internal structure.")

print(f"✅ Action Class path determined: {ACTION_CLASS_PATH}")


# --- CONSOLIDATION LOGIC ---
ACTION_CLASSES = sorted([d for d in os.listdir(ACTION_CLASS_PATH)
                         if os.path.isdir(os.path.join(ACTION_CLASS_PATH, d)) and len(d) > 1])
NUM_CLASSES = len(ACTION_CLASSES)
SEQUENCE_LENGTH = 40

# Check if we found the intended classes
if NUM_CLASSES < 3:
    raise ValueError(f"❌ FATAL ERROR: Found only {NUM_CLASSES} classes: {ACTION_CLASSES}. Data is severely incomplete. Expected at least 3 action classes.")

print(f"✅ Found {NUM_CLASSES} Action Classes: {ACTION_CLASSES}")

# --- FEATURE INDEXING (126 Features) ---
HAND_START_INDEX = 33 * 4 + 468 * 3
HAND_END_INDEX = HAND_START_INDEX + (21 * 3 * 2)
FINAL_FEATURE_SIZE = 126
print(f"Target Feature Vector Size: {FINAL_FEATURE_SIZE} (X,Y,Z for both hands)")

X_data = []
Y_labels = []
label_map = {word: i for i, word in enumerate(ACTION_CLASSES)}
corrupted_count = 0

# Loop 1: Iterate through each ACTION CLASS (e.g., 'hello')
for class_index, action_class in enumerate(tqdm(ACTION_CLASSES, desc="Consolidating Classes")):
    class_path = os.path.join(ACTION_CLASS_PATH, action_class)

    # Loop 2: Use os.walk to find the Trial/Sequence Folders (e.g., '97', '98')
    # This automatically handles the Trial/Sequence Folder level
    for root, dirs, files in os.walk(class_path):

        # Check if we are inside a Trial/Sequence Folder (meaning, we have .npy files)
        npy_files = sorted([f for f in files if f.endswith('.npy')])

        if len(npy_files) > 0 and len(npy_files) == SEQUENCE_LENGTH:
            # We found a complete sequence (Trial/Sequence Folder)

            sequence_data = []

            # Loop 3: Load the 40 .npy files chronologically (0.npy, 1.npy, ...)
            for i in range(SEQUENCE_LENGTH):
                file_name = f"{i}.npy"
                file_path = os.path.join(root, file_name)

                try:
                    frame_features = np.load(file_path)

                    # Ensure array is 1D (representing features for a single frame)
                    if frame_features.ndim != 1 or frame_features.shape[0] < HAND_END_INDEX:
                         raise ValueError("Incorrect feature shape or incomplete frame data.")

                    # Extract 126 features from the single frame vector
                    hand_landmarks_frame = frame_features[HAND_START_INDEX:HAND_END_INDEX]
                    sequence_data.append(hand_landmarks_frame)

                except Exception as e:
                    corrupted_count += 1
                    sequence_data = [] # Discard incomplete sequence
                    break # Break out of the frame loop, sequence is corrupt

            # If the sequence was loaded successfully (no breaks)
            if len(sequence_data) == SEQUENCE_LENGTH:
                X_data.append(np.array(sequence_data))
                Y_labels.append(class_index)


# Convert lists to NumPy arrays and One-Hot Encode Labels
X = np.array(X_data)
Y = np.array(Y_labels)
Y_one_hot = to_categorical(Y, num_classes=NUM_CLASSES)

# Print Final Shapes for Validation
print(f"\nSkipped {corrupted_count} corrupted frames (part of failed sequences).")
print(f"Total valid sequences extracted: {X.shape[0]}")
print("\n--- FINAL DATA SHAPES ---")
print(f"X (Sequences): {X.shape} (Samples, Timesteps, Features)")
print(f"Y (Labels):    {Y_one_hot.shape} (Samples, Classes)")

# --- DELIVERABLE: Save final NumPy arrays back to Drive ---
np.save(os.path.join(BASE_DRIVE_PATH, 'X_data_dynamic.npy'), X)
np.save(os.path.join(BASE_DRIVE_PATH, 'Y_data_dynamic.npy'), Y_one_hot)

print("✅ Data consolidation and filtering complete. Ready for LSTM Training.")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras.utils import to_categorical # Used if loading raw arrays later

# --- CONFIGURATION (Load Final Arrays) ---
BASE_DRIVE_PATH = '/content/drive/MyDrive/ASL_Project_Files'
os.chdir(BASE_DRIVE_PATH)

print("--- 1. LOADING FINAL DYNAMIC ARRAYS ---")

try:
    # Load the consolidated X and Y arrays saved from the previous step
    X = np.load('X_data_dynamic.npy')
    Y = np.load('Y_data_dynamic.npy')

except FileNotFoundError:
    print("❌ FATAL ERROR: Final dynamic arrays (X or Y) not found. Cannot proceed.")
    exit()

# Define final parameters from loaded data shapes
SEQUENCE_LENGTH = X.shape[1]    # Timesteps (e.g., 40 frames)
FINAL_FEATURE_SIZE = X.shape[2] # Features (126 features)
NUM_CLASSES = Y.shape[1]        # Classes (e.g., 5 words)

print(f"✅ Data loaded. Shape: {X.shape} (Samples, Timesteps, Features)")


# --- 2. PREDICTION TASK: LSTM MODEL DEFINITION AND TRAINING ---
print("\n--- 2. LSTM MODEL TRAINING AND EXPORT ---")

# Split data (80% Train, 20% Test)
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training Samples: {len(X_train)}; Test Samples: {len(X_test)}")

# --- MODEL DEFINITION (LSTM Architecture) ---
model = Sequential([
    # Masking layer is essential for sequence data (handles any zero-padding)
    Masking(mask_value=0., input_shape=(SEQUENCE_LENGTH, FINAL_FEATURE_SIZE)),

    # LSTM Layer: The core of the sequence processing model
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.4),

    # Dense output layer
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- TRAINING (The slow part with epochs) ---
print("\n--- TRAINING LSTM MODEL (This will take time due to sequence processing) ---")

history = model.fit(
    X_train, y_train_cat,
    epochs=25, # Standard epochs for sequence models
    batch_size=16, # Small batch size helps with sequence complexity
    validation_data=(X_test, y_test_cat),
    verbose=1
)

# --- EXPORT ---
MODEL_FILENAME_H5 = 'my_lstm_model_phase2.h5'
model.save(os.path.join(BASE_DRIVE_PATH, MODEL_FILENAME_H5))

# --- EVALUATION ---
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nFinal LSTM Model Accuracy: {accuracy:.4f}")
print(f"✅ Phase 2 Model saved to: {MODEL_FILENAME_H5}")

print("\n--- Phase 2 Prediction Task complete. ---")