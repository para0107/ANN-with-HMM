import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

from dataset import IAMDataset, CHARS, STATES_PER_CHAR, char_to_state_id
from model import ANN
from hmm import HybridHMM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'IAM', 'features')
XML_DIR = os.path.join(BASE_DIR, 'IAM', 'xml')

BATCH_SIZE = 1
LR = 0.0003
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, optimizer, criterion, hmm_decoder=None):
    model.train()
    total_loss = 0
    batches = 0

    for i, (features, targets, text) in enumerate(dataloader):
        features = features.to(DEVICE).squeeze(0)
        targets = targets.to(DEVICE).squeeze(0)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

        # --- SANITY CHECK: See results every 100 images ---
        if i % 100 == 0 and hmm_decoder is not None:
            # Quick decode to see what the model thinks
            with torch.no_grad():
                pred_text = hmm_decoder.decode(outputs.cpu().numpy())
            print(f"  [Batch {i}] Truth: '{text[0]}' | Pred: '{pred_text}'")
        # --------------------------------------------------

    return total_loss / batches if batches > 0 else 0


def validate(model, dataloader, hmm):
    model.eval()
    total_err = 0
    total_chars = 0

    with torch.no_grad():
        for features, _, text in dataloader:
            features = features.to(DEVICE).squeeze(0)
            outputs = model(features)
            pred = hmm.decode(outputs.cpu().numpy())

            # Simple Character Error calculation (Distance)
            # For strict CER you need Levenshtein distance,
            # but this is a rough proxy for valid/invalid
            if pred == text[0]:
                pass  # Match
            else:
                total_err += 1  # Count total wrong lines for now
            total_chars += 1

    return total_err / total_chars  # Returns Line Error Rate for simplicity


def main():
    print(f"Device: {DEVICE}")
    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # --- FIX: Exact calculation ---
    num_classes = len(CHARS) * STATES_PER_CHAR
    print(f"Classes: {num_classes} ({len(CHARS)} chars * {STATES_PER_CHAR} states)")

    model = ANN(num_classes=num_classes).to(DEVICE)
    hmm = HybridHMM(num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    # EM Loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")

        # E-Step: Alignment
        print("Aligning...")
        model.eval()
        hmm.reset_accumulators()
        for i in range(len(train_subset)):
            real_idx = train_subset.indices[i]
            feat, _, text = full_dataset.get_item_with_text(real_idx)
            feat = feat.to(DEVICE)
            with torch.no_grad():
                out = model(feat).cpu().numpy()

            scaled = hmm.get_scaled_emissions(out)

            # Build valid state sequence for this text
            state_seq = []
            for char in text:
                if char in CHARS:
                    base = char_to_state_id(char)
                    for s in range(STATES_PER_CHAR): state_seq.append(base + s)

            if len(state_seq) > 0:
                path = hmm.forced_alignment(scaled, state_seq)
                if path is not None:
                    full_dataset.update_target_at_index(real_idx, torch.from_numpy(path).long())

        hmm.update_parameters()

        # M-Step: Training
        print("Training...")
        loss = train_epoch(model, train_loader, optimizer, criterion, hmm_decoder=hmm)
        val_ler = validate(model, val_loader, hmm)
        print(f"Epoch {epoch} Done. Loss: {loss:.4f} | Val Line Error: {val_ler:.2%}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()