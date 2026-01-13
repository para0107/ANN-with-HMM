import torch
import cv2
import numpy as np
import argparse
import os

from model import ANN
from dataset import CHARS, TOTAL_STATES
from preprocess import clean_image, deslant_image, normalize_size, extract_features, deslope_image
from hmm import HybridHMM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best_model.pth")
WINDOW_WIDTH = 13  # Updated to 13


def load_model():
    print(f"Loading model from {WEIGHTS_PATH}...")
    # Architecture params must match training
    model = ANN(num_classes=TOTAL_STATES, window_width=WINDOW_WIDTH).to(DEVICE)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")

    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def prepare_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image.")

    img = clean_image(img)
    img = deslope_image(img)
    img = deslant_image(img)
    img = normalize_size(img)

    if img is None:
        raise ValueError("Image preprocessing failed.")

    features = extract_features(img)

    if len(features) == 0:
        raise ValueError("No features extracted.")

    half_window = WINDOW_WIDTH // 2
    feat_dim = features.shape[1]

    features_padded = np.pad(features, ((half_window, half_window), (0, 0)), mode='edge')

    num_frames = features.shape[0]
    windows = np.zeros((num_frames, WINDOW_WIDTH * feat_dim), dtype=np.float32)

    for t in range(num_frames):
        win = features_padded[t: t + WINDOW_WIDTH]
        windows[t] = win.flatten()

    mean = windows.mean()
    std = windows.std() + 1e-6
    windows = (windows - mean) / std

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
        outputs = model(input_tensor)
        log_probs = outputs.squeeze(0).cpu().numpy()
        decoded_text = hmm.decode(log_probs)

        print("\n" + "=" * 40)
        print(f"PREDICTION:  {decoded_text}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict handwritten text from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    predict(args.image)