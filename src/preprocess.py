import os
import cv2
import numpy as np

# --- Configuration ---
# 1. Get the absolute path of the directory where 'preprocess.py' sits
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to get the project root (where 'IAM' and 'src' both live)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 3. Define DATA_ROOT relative to that project root
DATA_ROOT = os.path.join(PROJECT_ROOT, "IAM")

# 4. Define the rest of your paths
LINES_FILE = os.path.join(DATA_ROOT, "ascii", "lines.txt")
IMAGE_DIR = os.path.join(DATA_ROOT, "data", "lines")
OUTPUT_DIR = os.path.join(DATA_ROOT, "features")

# Optional: Add a check to fail fast if the path is still wrong
if not os.path.exists(LINES_FILE):
    raise FileNotFoundError(f"Could not find lines.txt at: {LINES_FILE}")

# Paper specifications
TARGET_HEIGHT = 128
GRID_ROWS = 20
WINDOW_SIZE = 9


def clean_image(img):
    """
    Replicates 'Enhancer-MLP': Removes noise and binarizes.
    """
    # 1. Grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Invert (Ink should be white, background black)
    img = cv2.bitwise_not(img)

    # 3. Denoise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 4. Binarize (Otsu's method handles varying contrast)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img


def deslant_image(img):
    """
    Replicates 'Slant-MLP': Shears image to be upright.
    Uses Moments analysis to find principal axis.
    """
    h, w = img.shape
    moments = cv2.moments(img)

    if moments['mu02'] == 0:
        return img

    # Calculate skew
    skew = moments['mu11'] / moments['mu02']

    # Create affine transform matrix for shearing
    M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])

    # Apply shear
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return img


def normalize_size(img):
    """
    Replicates 'Slope-MLP' and 'Normalize-MLP'.
    Crops to the main ink area and resizes to fixed height.
    """
    # 1. Find bounding box of ink
    coords = cv2.findNonZero(img)
    if coords is None: return None  # Empty image

    x, y, w, h = cv2.boundingRect(coords)

    # 2. Crop
    img = img[y:y + h, x:x + w]

    # 3. Resize to fixed height, maintaining aspect ratio
    scale = TARGET_HEIGHT / h
    new_w = int(w * scale)

    # Avoid errors with tiny images
    if new_w <= 0: return None

    img = cv2.resize(img, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # 4. Re-binarize to keep edges sharp after resize
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    return img


def extract_features(img):
    """
    Replicates Paper Section 2.5: Grid-based feature extraction.
    Generates the (Time, 60) feature matrix.
    """
    h, w = img.shape

    # Normalize pixel values to 0-1 range
    img = img.astype(np.float32) / 255.0

    # Calculate cell height
    cell_h = h // GRID_ROWS

    features = []

    # Iterate over columns (Time steps)
    for x in range(w):
        col_feats = []

        # Calculate gradients (derivatives) for this column
        if x > 0 and x < w - 1:
            h_deriv = img[:, x + 1] - img[:, x - 1]
        else:
            h_deriv = np.zeros(h)

        v_deriv = np.zeros(h)
        v_deriv[1:-1] = img[2:, x] - img[:-2, x]

        # Iterate over 20 vertical cells
        for r in range(GRID_ROWS):
            y_start = r * cell_h
            y_end = (r + 1) * cell_h

            # If height doesn't divide perfectly, handle edge case
            if y_end > h: y_end = h

            # 1. Normalized Gray Level
            val_n = np.mean(img[y_start:y_end, x])

            # 2. Horizontal Derivative
            val_h = np.mean(h_deriv[y_start:y_end])

            # 3. Vertical Derivative
            val_v = np.mean(v_deriv[y_start:y_end])

            col_feats.extend([val_n, val_h, val_v])

        features.append(col_feats)

    return np.array(features, dtype=np.float32)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Reading {LINES_FILE}...")

    with open(LINES_FILE, 'r') as f:
        lines = f.readlines()

    processed_count = 0

    for line in lines:
        # Skip comments
        if line.startswith("#"): continue

        parts = line.strip().split()

        # Check segmentation status
        status = parts[1]
        if status == "err":
            continue  # Skip bad lines

        # Parse ID and Text
        line_id = parts[0]  # e.g. "a01-000u-00"

        # Text is the last part, separated by |
        # In lines.txt, content starts at index 8
        label_raw = parts[8]
        label_clean = label_raw.replace("|", " ")

        # Construct Image Path
        # ID: a01-000u-00 -> a01 / a01-000u / a01-000u-00.png
        root_folder = line_id.split("-")[0]
        sub_folder = f"{root_folder}-{line_id.split('-')[1]}"
        img_path = os.path.join(IMAGE_DIR, root_folder, sub_folder, line_id + ".png")

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        try:
            # --- The Pipeline ---
            # 1. Load
            img = cv2.imread(img_path)
            if img is None: continue

            # 2. Preprocess (Geometric)
            img = clean_image(img)
            img = deslant_image(img)
            img = normalize_size(img)

            if img is None: continue  # Skip if image became invalid

            # 3. Feature Extraction
            feats = extract_features(img)

            # 4. Save
            out_name = line_id + ".npy"
            np.save(os.path.join(OUTPUT_DIR, out_name), feats)

            # Optional: Save the text label to a separate file if you want
            # (We will usually read it from lines.txt during training though)

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} lines...")

        except Exception as e:
            print(f"Error processing {line_id}: {e}")

    print(f"Done. Successfully processed {processed_count} lines.")
    print(f"Features saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()