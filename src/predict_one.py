import torch
import cv2
import numpy as np
import argparse
import os

# Import components from your existing project files
from model import ANN
from dataset import TOTAL_STATES
from hmm import HybridHMM
# We import specific functions from preprocess to ensure exact consistency with training
from preprocess import clean_image, deslope_image, deslant_image, normalize_size, extract_features

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Assumes you run this from the 'src' folder or project root. Adjust if needed.
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best_model.pth")
WINDOW_WIDTH = 9


def load_model():
    print(f"Loading model from {WEIGHTS_PATH}...")
    model = ANN(num_classes=TOTAL_STATES).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}. Please train the model first.")

    # Load weights
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def prepare_image(image_path):
    """
    Reads an image and applies the exact same pipeline as the training loader.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Processing {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image (corrupted or unsupported format).")

    # 1. Visual Preprocessing (Same as preprocess.py)
    img = clean_image(img)
    img = deslope_image(img)
    img = deslant_image(img)
    img = normalize_size(img)

    if img is None or img.size == 0:
        raise ValueError("Image preprocessing failed (image became empty).")

    # 2. Feature Extraction
    features = extract_features(img)  # Returns (Time, 60)

    if len(features) == 0:
        raise ValueError("No features extracted from image.")

    # 3. Windowing (Same as dataset.py __getitem__)
    # We must slide a window of size 9 over the features
    half_window = WINDOW_WIDTH // 2
    feat_dim = features.shape[1]

    # Pad features so the window can center on the edge frames
    features_padded = np.pad(features, ((half_window, half_window), (0, 0)), mode='edge')

    num_frames = features.shape[0]
    windows = np.zeros((num_frames, WINDOW_WIDTH * feat_dim), dtype=np.float32)

    for t in range(num_frames):
        win = features_padded[t: t + WINDOW_WIDTH]
        windows[t] = win.flatten()

    # 4. Instance Normalization (Crucial!)
    # The training loader normalizes each sample by its own mean/std.
    mean = windows.mean()
    std = windows.std() + 1e-6
    windows = (windows - mean) / std

    # Convert to Tensor (Batch, Time, Feats)
    tensor = torch.from_numpy(windows).unsqueeze(0).to(DEVICE)
    return tensor


def predict(image_path):
    model = load_model()
    hmm = HybridHMM(num_classes=TOTAL_STATES)

    try:
        input_tensor = prepare_image(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    with torch.no_grad():
        # Get neural network output (Log Softmax)
        outputs = model(input_tensor)
        log_probs = outputs.squeeze(0).cpu().numpy()

        # Decode using HMM (Viterbi)
        decoded_text = hmm.decode(log_probs)

        print("\n" + "=" * 40)
        print(f"PREDICTION:  {decoded_text}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict handwritten text from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    predict(args.image)