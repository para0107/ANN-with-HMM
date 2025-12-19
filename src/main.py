import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

# Custom modules
from dataset import IAMDataset, CHARS, STATES_PER_CHAR, char_to_state_id
from model import ANN
from hmm import HybridHMM
from metrics import calculate_error_rates, greedy_decode, plot_training_history

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'IAM', 'features')
XML_DIR = os.path.join(BASE_DIR, 'IAM', 'xml')

# --- Config ---
BATCH_SIZE = 1
LR = 0.0003
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    batches = 0

    for features, targets, _ in dataloader:
        features = features.to(DEVICE).squeeze(0)  # (T, 540)
        targets = targets.to(DEVICE).squeeze(0)  # (T,)

        optimizer.zero_grad()
        outputs = model(features)  # Should be (T, num_classes)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1
    return total_loss / batches if batches > 0 else 0


def validate(model, dataloader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for features, _, text in dataloader:
            features = features.to(DEVICE).squeeze(0)
            outputs = model(features)

            # Greedy Decode
            pred_str = greedy_decode(outputs.cpu().numpy())
            preds.append(pred_str)
            truths.append(text[0])

    return calculate_error_rates(preds, truths)


def main():
    print(f"Using Device: {DEVICE}")
    print(f"Loading data from: {FEATURE_DIR}")

    # 1. Load Dataset
    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)

    # 2. Split Data
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_subset)} | Val samples: {len(val_subset)}")

    # --- CRITICAL FIX START ---
    # Calculate exactly how many output neurons we need.
    # Dataset labels go from 0 to (num_chars * states_per_char) - 1.
    # If the max label seen is 546, we need 547 outputs.
    num_classes = (len(CHARS) * STATES_PER_CHAR) + 1
    print(f"Dataset requires {num_classes} output neurons (based on {len(CHARS)} chars * {STATES_PER_CHAR} states).")

    # Pass this number to the ANN
    model = ANN(num_classes=num_classes).to(DEVICE)
    # --- CRITICAL FIX END ---

    hmm = HybridHMM(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    history = {'loss': [], 'cer': [], 'wer': []}

    # --- Epoch 0: Flat Start ---
    print("\n--- Epoch 0 (Flat Start) ---")
    loss = train_epoch(model, train_loader, optimizer, criterion)
    cer, wer = validate(model, val_loader)
    print(f"Loss: {loss:.4f} | CER: {cer:.2%} | WER: {wer:.2%}")
    history['loss'].append(loss)
    history['cer'].append(cer)
    history['wer'].append(wer)

    # --- EM Loop ---
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch} (EM Cycle) ---")

        # A. ALIGNMENT (E-Step)
        print("Aligning training data...")
        model.eval()
        hmm.reset_accumulators()

        for i in range(len(train_subset)):
            real_idx = train_subset.indices[i]
            feat, _, text = full_dataset.get_item_with_text(real_idx)
            feat = feat.to(DEVICE)

            with torch.no_grad():
                out = model(feat).cpu().numpy()

            scaled = hmm.get_scaled_emissions(out)

            state_seq = []
            for char in text:
                if char in CHARS:
                    base = char_to_state_id(char)
                    for s in range(STATES_PER_CHAR): state_seq.append(base + s)

            # Only align if we have a valid sequence
            if len(state_seq) > 0:
                path = hmm.forced_alignment(scaled, state_seq)
                if path is not None:
                    full_dataset.update_target_at_index(real_idx, torch.from_numpy(path).long())

        hmm.update_parameters()

        # B. TRAINING (M-Step)
        loss = train_epoch(model, train_loader, optimizer, criterion)
        cer, wer = validate(model, val_loader)

        print(f"Loss: {loss:.4f} | CER: {cer:.2%} | WER: {wer:.2%}")
        history['loss'].append(loss)
        history['cer'].append(cer)
        history['wer'].append(wer)

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    plot_training_history(history, "training_plot.png")


if __name__ == "__main__":
    main()