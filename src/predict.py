import torch
import cv2
import numpy as np
import argparse
import os

from src.model import ANN
from src.dataset import CHARS, TOTAL_STATES
from src.preprocess import clean_image, deslant_image, normalize_size, extract_features, deslope_image
from src.hmm import HybridHMM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_epoch_5.pth"


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

    img = clean_image(img)
    img = deslope_image(img)
    img = deslant_image(img)
    img = normalize_size(img)

    if img is None:
        print("Error: Image preprocessing failed (empty or invalid).")
        return None

    features = extract_features(img)

    if features.shape[0] == 0:
        print("Error: Extracted features are empty.")
        return None


    window_width = 9
    half_window = window_width // 2
    feat_dim = features.shape[1]

    features_padded = np.pad(features, ((half_window, half_window), (0, 0)), mode='edge')
    num_frames = features.shape[0]

    windows = np.zeros((num_frames, window_width * feat_dim), dtype=np.float32)
    for t in range(num_frames):
        win = features_padded[t: t + window_width]
        windows[t] = win.flatten()

    tensor = torch.from_numpy(windows).to(DEVICE)
    return tensor


def predict(image_path):

    num_classes = TOTAL_STATES
    print(f"Initializing model with {num_classes} states...")

    model = ANN(num_classes=num_classes).to(DEVICE)

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

    input_tensor = process_single_image(image_path)
    if input_tensor is None: return

    with torch.no_grad():

        outputs = model(input_tensor)

        log_probs = outputs.cpu().numpy()

        text = hmm.decode(log_probs)

        print("\n" + "=" * 30)
        print(f"PREDICTION: {text}")
        print("=" * 30 + "\n")


if __name__ == "__main__":
    target_image = "path/to/your/image.png"
    predict(target_image)