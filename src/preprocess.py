import os
import cv2
import numpy as np

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "IAM")
LINES_FILE = os.path.join(DATA_ROOT, "ascii", "lines.txt")
IMAGE_DIR = os.path.join(DATA_ROOT, "data", "lines")
OUTPUT_DIR = os.path.join(DATA_ROOT, "features")

if not os.path.exists(LINES_FILE):
    pass  # Non-critical if running inference only

TARGET_HEIGHT = 128
GRID_ROWS = 20
WINDOW_SIZE = 9


def clean_image(img):
    """Replicates 'Enhancer-MLP': Removes noise and binarizes."""
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)  # Invert: Ink=White
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def deslope_image(img):
    """
    Replicates 'Slope-MLP': Rotates the image so the text baseline is horizontal.
    Uses linear regression on ink pixels.
    """
    coords = np.column_stack(np.where(img > 0))
    if len(coords) < 50: return img  # Too little ink to judge slope

    # Coordinates are (y, x). Fit y = mx + c
    y, x = coords[:, 0], coords[:, 1]

    # Simple linear regression
    m, c = np.polyfit(x, y, 1)
    angle = np.arctan(m) * (180 / np.pi)

    # Ignore extreme angles (likely vertical lines or noise)
    if abs(angle) > 20: return img

    # Rotate
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img


def deslant_image(img):
    """Replicates 'Slant-MLP': Shears image to be upright."""
    h, w = img.shape
    moments = cv2.moments(img)
    if moments['mu02'] == 0: return img
    skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img


def normalize_size(img):
    """Replicates 'Normalize-MLP'."""
    coords = cv2.findNonZero(img)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y + h, x:x + w]

    scale = TARGET_HEIGHT / h
    new_w = int(w * scale)
    if new_w <= 0: return None

    img = cv2.resize(img, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img


def extract_features(img):
    """Replicates Section 2.5: Grid-based feature extraction."""
    h, w = img.shape
    img = img.astype(np.float32) / 255.0
    cell_h = h // GRID_ROWS
    features = []

    for x in range(w):
        col_feats = []
        if x > 0 and x < w - 1:
            h_deriv = img[:, x + 1] - img[:, x - 1]
        else:
            h_deriv = np.zeros(h)
        v_deriv = np.zeros(h)
        v_deriv[1:-1] = img[2:, x] - img[:-2, x]

        for r in range(GRID_ROWS):
            y_start = r * cell_h
            y_end = (r + 1) * cell_h
            if y_end > h: y_end = h

            val_n = np.mean(img[y_start:y_end, x])
            val_h = np.mean(h_deriv[y_start:y_end])
            val_v = np.mean(v_deriv[y_start:y_end])
            col_feats.extend([val_n, val_h, val_v])
        features.append(col_feats)
    return np.array(features, dtype=np.float32)


def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"Reading {LINES_FILE}...")
    with open(LINES_FILE, 'r') as f:
        lines = f.readlines()

    processed_count = 0
    for line in lines:
        if line.startswith("#"): continue
        parts = line.strip().split()
        if parts[1] == "err": continue
        line_id = parts[0]

        root_folder = line_id.split("-")[0]
        sub_folder = f"{root_folder}-{line_id.split('-')[1]}"
        img_path = os.path.join(IMAGE_DIR, root_folder, sub_folder, line_id + ".png")

        if not os.path.exists(img_path): continue

        try:
            img = cv2.imread(img_path)
            if img is None: continue

            # --- UPDATED PIPELINE ---
            img = clean_image(img)
            img = deslope_image(img)  # <--- NEW STEP
            img = deslant_image(img)
            img = normalize_size(img)

            if img is None: continue
            feats = extract_features(img)
            np.save(os.path.join(OUTPUT_DIR, line_id + ".npy"), feats)

            processed_count += 1
            if processed_count % 100 == 0: print(f"Processed {processed_count} lines...")
        except Exception as e:
            print(f"Error {line_id}: {e}")


if __name__ == "__main__":
    main()