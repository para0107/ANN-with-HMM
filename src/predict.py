import torch
import cv2
import numpy as np
import argparse
import os

# Import your modules
from src.model import ANN
# UPDATED: Import TOTAL_STATES
from src.dataset import CHARS, TOTAL_STATES
from src.preprocess import clean_image, deslant_image, normalize_size, extract_features, deslope_image
from src.hmm import HybridHMM

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_epoch_5.pth"  # Change this to your latest model file


def process_single_image(image_path):
    """
    Applies the exact same preprocessing as training.
    """
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return None

    # Pipeline (Make sure this matches preprocess.py!)
    img = clean_image(img)
    img = deslope_image(img)  # Added this step
    img = deslant_image(img)
    img = normalize_size(img)

    if img is None:
        print("Error: Image preprocessing failed (empty or invalid).")
        return None

    features = extract_features(img)

    # Check if features are empty
    if features.shape[0] == 0:
        print("Error: Extracted features are empty.")
        return None

    # The ANN expects (Batch, Input_Size).
    # Since dataset.py pads/windows the data, we must do the same here.
    window_width = 9
    half_window = window_width // 2
    feat_dim = features.shape[1]  # 60

    # Pad features to handle borders
    features_padded = np.pad(features, ((half_window, half_window), (0, 0)), mode='edge')
    num_frames = features.shape[0]

    windows = np.zeros((num_frames, window_width * feat_dim), dtype=np.float32)
    for t in range(num_frames):
        win = features_padded[t: t + window_width]
        windows[t] = win.flatten()

    # Convert to tensor: (Time_Steps, 540)
    tensor = torch.from_numpy(windows).to(DEVICE)
    return tensor


def predict(image_path):
    # 1. Setup Model
    # UPDATED: Use TOTAL_STATES directly
    num_classes = TOTAL_STATES
    print(f"Initializing model with {num_classes} states...")

    model = ANN(num_classes=num_classes).to(DEVICE)

    # Load Weights
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Loaded weights from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Ensure that the model architecture matches the checkpoint (TOTAL_STATES).")
            return
    else:
        print(f"Warning: {MODEL_PATH} not found. Using random weights (Garbage output expected).")

    model.eval()
    hmm = HybridHMM(num_classes=num_classes)

    # 2. Process Image
    input_tensor = process_single_image(image_path)
    if input_tensor is None: return

    # 3. Inference
    with torch.no_grad():
        # Pass data. The model expects (Batch, 540).
        # We treat Time steps as a "Batch" of frames for the forward pass.
        outputs = model(input_tensor)  # Output: (Time, Num_Classes)

        # Convert to numpy
        log_probs = outputs.cpu().numpy()

        # Decode using the HMM's greedy decoder
        text = hmm.decode(log_probs)

        print("\n" + "=" * 30)
        print(f"PREDICTION: {text}")
        print("=" * 30 + "\n")


if __name__ == "__main__":
    # Change this to your target image
    target_image = "path/to/your/image.png"
    predict(target_image)