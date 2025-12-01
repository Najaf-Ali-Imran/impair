!pip install pandas scikit-learn joblib seaborn matplotlib

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION (NO CHANGES HERE IF YOUR PATH IS ALREADY CORRECT) ---
BASE_DIR = '/content/drive/MyDrive/ASL_Project_Files'
DATASET_PATH = os.path.join(BASE_DIR, 'hand_keypoints.csv')

try:
    os.chdir(BASE_DIR)
except FileNotFoundError:
    print(f"❌ FATAL ERROR: Project Directory not found at {BASE_DIR}.")
    exit()

print("--- 1. DATA LOADING & VALIDATION (CORRECTION APPLIED) ---")

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset loaded successfully from: {DATASET_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: CSV file 'hand_keypoints.csv' not found inside the {BASE_DIR} folder.")
    exit()

# === CRITICAL FIXES FOR THIS DATASET STRUCTURE ===
# 1. Manually set the correct label column
LABEL_COLUMN = 'folder'

# 2. Identify and drop non-feature columns (filename)
df.drop(columns=['filename'], inplace=True)

# 3. Redefine Features and check count
FEATURE_COLUMNS = df.columns.drop(['folder', 'y20']) # Drop folder (label) and y20 (extra)

# Re-check the column definitions based on the corrected DataFrame
TOTAL_COLUMNS = len(df.columns)
if TOTAL_COLUMNS == 43: # 42 Features (x0-y20) + 1 Label (folder)
    print("✅ Columns re-parsed successfully.")

else: # If the original CSV has 44 columns: folder, filename, x0-y20
    # The columns are likely: folder, x0, y0, x1, y1, ..., x20, y20, plus another header column.
    # We will assume the structure is: folder, x0, y0, ..., y20. (Total 44 in original CSV output)
    # The initial read showed 44 columns, but pandas reports 43 features. Let's inspect again.

    # We will assume the features are everything NOT 'folder' and NOT 'filename'
    FEATURE_COLUMNS = df.columns.drop(LABEL_COLUMN)
    print(f"⚠️ Warning: Total columns is {len(df.columns)}. Reverting Feature column definition.")

# --- RE-DISPLAY RESULTS WITH CORRECTED COLUMNS ---

print(f"\nTotal Samples (Rows): {len(df)}")
print(f"Total Features (Columns): {len(FEATURE_COLUMNS)} (Target: 42 features)")
print(f"Label Column Name: {LABEL_COLUMN} (CORRECTED)")
print(f"First 5 Rows (Features only):\n{df[FEATURE_COLUMNS].head()}")

# Final Feature Check
if len(FEATURE_COLUMNS) == 42:
    print("✅ Feature count is NOW correct (42 features).")
else:
    print(f"❌ Final Feature count is {len(FEATURE_COLUMNS)}. This needs manual confirmation.")


# Class Analysis (Now checking the correct 'folder' column)
class_counts = df[LABEL_COLUMN].value_counts().sort_index()
all_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

print("-" * 40)
print(f"Total Classes Found: {len(class_counts)}")
print(f"Classes Present: {class_counts.index.tolist()}")
print(f"Missing Letters: {sorted(list(set(all_letters) - set(class_counts.index.tolist())))}")

print("\n--- Summary: Class Distribution (Top 10) ---")
print(class_counts.head(10))


from sklearn.preprocessing import StandardScaler
import joblib # Already imported in the final runnable script

print("\n--- 2. DATA CLEANING & PREPROCESSING ---")

# --- Define Final Columns Based on Manual Inspection ---
LABEL_COLUMN = 'folder'
METADATA_COLS = ['filename', 'folder']
# Features are all columns EXCEPT 'filename' and 'folder'
FEATURE_COLUMNS = [col for col in df.columns if col not in METADATA_COLS]

# 1. Handle Duplicates and Missing Values (Strategy: Drop rows)
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"Removed {initial_rows - len(df)} duplicate rows.")

rows_before_na = len(df)
df.dropna(inplace=True)
print(f"Removed {rows_before_na - len(df)} rows with missing values (NaNs).")
print(f"Cleaned Samples: {len(df)}")
df_cleaned = df.copy() # This copy will be saved as the cleaned deliverable.

# 2. Label Encoding (Encoding 'A' -> 0, 'B' -> 1, etc.)
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[LABEL_COLUMN])
print(f"Labels encoded (Example: '{le.classes_[0]}' is mapped to {df['label_encoded'].iloc[0]}).")

# 3. Scaling/Normalization (Strategy: Standard Scaling)
# Justification: Standardizing coordinates improves the performance of distance-based models (KNN, LR).
scaler = StandardScaler()
df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
print("✅ Features standardized (mean=0, std=1).")

# --- DELIVERABLE 6: Save Cleaned CSV ---
df_cleaned.to_csv(os.path.join(BASE_DIR, 'ASL_cleaned_dataset.csv'), index=False)
print("✅ Saved cleaned CSV: ASL_cleaned_dataset.csv")

# Save the scaler and encoder (CRITICAL for Person B)
joblib.dump(scaler, os.path.join(BASE_DIR, 'feature_scaler.pkl'))
joblib.dump(le, os.path.join(BASE_DIR, 'label_encoder.pkl'))
print("✅ Saved scaler and label encoder for integration.")

# --- Option: Subset for Testing ---
USE_SUBSET = True
if USE_SUBSET:
    test_subset = ['A', 'B', 'C', 'D', 'E']
    df_train = df[df[LABEL_COLUMN].isin(test_subset)].copy()
    print(f"⚠️ Training on subset A-E only. Samples: {len(df_train)}")
else:
    df_train = df.copy()
    print(f"Training on full dataset. Samples: {len(df_train)}")


import pandas as pd
import joblib
import os
import numpy as np

# --- CONFIGURATION (Ensure BASE_DIR is correct) ---
BASE_DIR = '/content/drive/MyDrive/ASL_Project_Files'
os.chdir(BASE_DIR)

print("--- VERIFYING SAVED DELIVERABLES ---")

# 1. Verify Cleaned CSV (ASL_cleaned_dataset.csv)
try:
    df_check = pd.read_csv('ASL_cleaned_dataset.csv')
    print(f"\n✅ ASL_cleaned_dataset.csv loaded.")
    print(f"   Shape: {df_check.shape} (Expected: ~10473 rows, 43 columns)")
    print(f"   Check for NaNs: {df_check.isnull().sum().sum()} (Expected: 0)")
    print("   Head (First 5 rows):\n", df_check.head())
except Exception as e:
    print(f"❌ ERROR loading ASL_cleaned_dataset.csv: {e}")

# 2. Verify Feature Scaler (.pkl)
try:
    scaler_check = joblib.load('feature_scaler.pkl')
    print(f"\n✅ Feature Scaler (feature_scaler.pkl) loaded.")
    # The scaler must have learned 41 features.
    print(f"   Total features learned by scaler: {scaler_check.mean_.shape[0]}")
    print(f"   Check mean for first feature (x0): {scaler_check.mean_[0]:.4f}")

    # Test the scaler: should be close to zero mean, unit variance
    # This checks if the StandardScaler object itself was saved correctly.
    dummy_input = df_check.iloc[0:1, 1:-1] # Take first row of features
    scaled_output = scaler_check.transform(dummy_input)
    print(f"   Test Scaling (Input vs. Scaled[0]): {dummy_input.iloc[0, 0]:.4f} -> {scaled_output[0, 0]:.4f}")

except Exception as e:
    print(f"❌ ERROR loading feature_scaler.pkl: {e}")

# 3. Verify Label Encoder (.pkl)
try:
    encoder_check = joblib.load('label_encoder.pkl')
    print(f"\n✅ Label Encoder (label_encoder.pkl) loaded.")
    print(f"   Classes present: {encoder_check.classes_.tolist()}")
    print(f"   Test decode (0 -> A): {encoder_check.inverse_transform([0])[0]}")

except Exception as e:
    print(f"❌ ERROR loading label_encoder.pkl: {e}")


# Continue in the next Colab cell
import matplotlib.pyplot as plt
import seaborn as sns

# --- Define Final Columns (Re-confirmed from successful load) ---
LABEL_COLUMN = 'folder'
METADATA_COLS = ['filename', 'folder', 'label_encoded']
FEATURE_COLUMNS = [col for col in df_train.columns if col not in METADATA_COLS]
# FEATURE_COLUMNS now contains exactly 42 coordinates.

print("\n--- 3. DATA INSIGHTS & VISUALIZATION ---")

# 1. Class Distribution Bar Chart (using the current subset A-E)
plt.figure(figsize=(8, 5))
sns.countplot(x=LABEL_COLUMN, data=df_train, order=df_train[LABEL_COLUMN].value_counts().index, palette='viridis')
plt.title('Class Distribution in Training Subset (A-E) 📊')
plt.xlabel('ASL Letter')
plt.ylabel('Number of Samples')
plt.grid(axis='y', linestyle='--')
plt.savefig(os.path.join(BASE_DIR, 'class_distribution.png'))
print("✅ Saved: class_distribution.png")


# 2. Sample Hand Poses (Scatter Plot)
def plot_hand_pose(row, title):
    """Plots 21 (x,y) points, simulating a hand pose from scaled coordinates."""
    # Features are x0, y0, x1, y1, ... x20, y20 (42 features)
    x_coords = row.filter(regex='^x').values
    y_coords = row.filter(regex='^y').values

    # Define connections (simplified based on MediaPipe hand structure)
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    plt.figure(figsize=(4, 4))
    plt.scatter(x_coords, y_coords, marker='o', s=50, c='red')

    # Plot connecting lines
    for start, end in CONNECTIONS:
        if start < len(x_coords) and end < len(x_coords):
            plt.plot([x_coords[start], x_coords[end]],
                     [y_coords[start], y_coords[end]], 'b-', alpha=0.6)

    plt.gca().invert_yaxis()
    plt.title(f"Pose: {title}")
    plt.xlabel('X (Scaled)')
    plt.ylabel('Y (Scaled)')
    plt.grid(True, linestyle=':')

# Select and plot 3 random samples from the subset
samples = df_train.sample(3, random_state=42)
for i, (index, row) in enumerate(samples.iterrows()):
    plot_hand_pose(row, f"Letter {row[LABEL_COLUMN]}")
    plt.savefig(os.path.join(BASE_DIR, f'sample_pose_{row[LABEL_COLUMN]}_{i}.png'))
    # plt.show()
print("✅ Saved: 3 sample hand pose PNGs.")
print("\nVisualization step complete.")