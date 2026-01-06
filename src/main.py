import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

from dataset import IAMDataset, CHARS, STATES_PER_CHAR, char_to_state_id
from model import ANN
from hmm import HybridHMM
from metrics import calculate_cer

# NEW IMPORT
from debug_utils import raw_dump

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'IAM', 'features')
XML_DIR = os.path.join(BASE_DIR, 'IAM', 'xml')

BATCH_SIZE = 1
LR = 0.0002
EPOCHS = 20
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batches += 1

        if i % 100 == 0:
            # 1. HMM Decode
            with torch.no_grad():
                hmm_pred = hmm_decoder.decode(outputs.cpu().numpy())

            # 2. Raw Network Output (What the brain sees before HMM logic)
            raw_pred = raw_dump(outputs.unsqueeze(0))

            print(f"  [Batch {i}] Truth: '{text[0]}'")
            print(f"             Raw:   '{raw_pred}'")
            print(f"             HMM:   '{hmm_pred}'")
            print("-" * 20)

    return total_loss / batches if batches > 0 else 0


def validate(model, dataloader, hmm):
    model.eval()
    total_cer = 0
    total_lines = 0
    with torch.no_grad():
        for features, _, text in dataloader:
            features = features.to(DEVICE).squeeze(0)
            outputs = model(features)
            pred = hmm.decode(outputs.cpu().numpy())
            cer = calculate_cer(pred, text[0])
            total_cer += cer
            total_lines += 1
    return total_cer / total_lines if total_lines > 0 else 0


def main():
    print(f"Device: {DEVICE}")
    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(CHARS) * STATES_PER_CHAR
    print(f"Classes: {num_classes} ({len(CHARS)} chars * {STATES_PER_CHAR} states)")

    model = ANN(num_classes=num_classes).to(DEVICE)
    hmm = HybridHMM(num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")

        # --- AGGRESSIVE STRATEGY: NO WARMUP ---
        # We align from Epoch 1. It will be messy but better than Flat Start poisoning.
        print("Aligning (Viterbi)...")
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

            if len(state_seq) > 0:
                path = hmm.forced_alignment(scaled, state_seq)
                if path is not None:
                    full_dataset.update_target_at_index(real_idx, torch.from_numpy(path).long())
        hmm.update_parameters()

        print("Training...")
        loss = train_epoch(model, train_loader, optimizer, criterion, hmm_decoder=hmm)
        avg_cer = validate(model, val_loader, hmm)

        print(f"Epoch {epoch} Done. Loss: {loss:.4f} | Avg CER: {avg_cer:.2%}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()