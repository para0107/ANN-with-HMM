import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# Custom modules
# UPDATED IMPORTS: Removed STATES_PER_CHAR, added TOTAL_STATES
from dataset import IAMDataset, CHARS, TOTAL_STATES
from model import ANN
from hmm import HybridHMM
from debug_utils import raw_dump

# --- Config ---
# We use a very small learning rate to be safe, but high enough to learn one image
LR = 0.001
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'IAM', 'features')
XML_DIR = os.path.join(BASE_DIR, 'IAM', 'xml')


def main():
    print(f"--- DEBUG MODE: Overfitting One Image ---")
    print(f"Using Device: {DEVICE}")

    # 1. Load Dataset
    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)

    # 2. Pick ONLY the first image that has text
    if len(full_dataset) == 0:
        print("Error: No data found.")
        return

    # Create a subset of just index 0
    debug_subset = Subset(full_dataset, [0])
    train_loader = DataLoader(debug_subset, batch_size=1, shuffle=False)

    # Get the text of this image for reference
    _, _, sample_text = full_dataset[0]
    print(f"Target Text to Learn: '{sample_text}'")

    # 3. Model Setup
    # UPDATED: Use TOTAL_STATES directly
    num_classes = TOTAL_STATES
    print(f"Initializing for {num_classes} dynamic states.")

    model = ANN(num_classes=num_classes).to(DEVICE)
    hmm = HybridHMM(num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    model.train()

    # 4. Training Loop (No Re-Alignment, just Flat Start)
    print("\nStarting Training (Expect Loss to drop to near 0)...")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0

        for features, targets, text in train_loader:
            features = features.to(DEVICE).squeeze(0)  # (T, 540)
            targets = targets.to(DEVICE).squeeze(0)  # (T,)

            optimizer.zero_grad()
            outputs = model(features)  # (T, Classes)

            # Important: We use the "Flat Start" targets provided by dataset.__getitem__
            # We are NOT re-aligning with HMM here. We want to force the NN to learn
            # the approximate locations first.

            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient Clipping (Safety)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

            # Sanity Check Print every 10 epochs
            if epoch % 10 == 0:
                with torch.no_grad():
                    # Check raw output using the new debug util
                    raw_str = raw_dump(outputs.unsqueeze(0))
                    # Check HMM decode (should improve as well)
                    hmm_str = hmm.decode(outputs.cpu().numpy())

                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
                print(f"  Raw: '{raw_str}'")
                print(f"  HMM: '{hmm_str}'")

    print("\n--- Debug Finished ---")
    print("If 'HMM' matches 'Target Text', the model works.")


if __name__ == "__main__":
    main()