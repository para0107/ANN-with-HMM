import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

# Custom modules
# Added TOTAL_STATES and char_to_state_seq to handle dynamic topology
from dataset import IAMDataset, CHARS, TOTAL_STATES, char_to_state_seq
from model import ANN
from hmm import HybridHMM
from metrics import calculate_cer
from debug_utils import raw_dump

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'IAM', 'features')
XML_DIR = os.path.join(BASE_DIR, 'IAM', 'xml')

# --- Config ---
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, optimizer, criterion, hmm_decoder=None):
    model.train()
    total_loss = 0
    batches = 0

    for i, (features, targets, text) in enumerate(dataloader):
        features = features.to(DEVICE).squeeze(0)  # (T, 540)
        targets = targets.to(DEVICE).squeeze(0)  # (T,)

        optimizer.zero_grad()
        outputs = model(features)  # (T, TOTAL_STATES)

        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        batches += 1

        # --- Monitoring ---
        if i % 100 == 0:
            # 1. HMM Decode (What the system thinks the text is)
            with torch.no_grad():
                hmm_pred = hmm_decoder.decode(outputs.cpu().numpy()) if hmm_decoder else "?"

            # 2. Raw Network Output (What the ANN sees before HMM logic)
            # This helps debug if the ANN is just predicting "Space" everywhere
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

            # Decode using HMM (Greedy)
            pred = hmm.decode(outputs.cpu().numpy())

            # Calculate Character Error Rate
            cer = calculate_cer(pred, text[0])
            total_cer += cer
            total_lines += 1

    return total_cer / total_lines if total_lines > 0 else 0


def main():
    print(f"Device: {DEVICE}")
    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)

    # 90/10 Train/Val Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # --- MODEL INITIALIZATION ---
    # Use TOTAL_STATES from dataset.py (Dynamic Topology)
    print(f"Initializing Model with {TOTAL_STATES} Total States.")
    model = ANN(num_classes=TOTAL_STATES).to(DEVICE)
    hmm = HybridHMM(num_classes=TOTAL_STATES)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")

        # --- ALIGNMENT PHASE (E-Step) ---
        # We start aligning aggressively after Epoch 1 to avoid "Flat Start" poisoning
        if epoch > 1:
            print("Aligning (Viterbi)...")
            model.eval()
            hmm.reset_accumulators()

            for i in range(len(train_subset)):
                real_idx = train_subset.indices[i]
                feat, _, text = full_dataset.get_item_with_text(real_idx)
                feat = feat.to(DEVICE)

                with torch.no_grad():
                    out = model(feat).cpu().numpy()

                # Get probabilities scaled by priors
                scaled = hmm.get_scaled_emissions(out)

                # Construct the specific state sequence for this text
                # Uses dynamic state counts (e.g. 'm' -> 5 states)
                state_seq = []
                for char in text:
                    # Append the specific list of states for this character
                    state_seq.extend(char_to_state_seq(char))

                # Viterbi Alignment
                if len(state_seq) > 0:
                    path = hmm.forced_alignment(scaled, state_seq)
                    if path is not None:
                        # Update the dataset target with the new aligned path
                        full_dataset.update_target_at_index(real_idx, torch.from_numpy(path).long())

            # Update HMM Transition/Prior probabilities based on alignment stats
            hmm.update_parameters()
        else:
            print("Skipping Alignment (Warmup Phase: Standard Flat Start)")

        # --- TRAINING PHASE (M-Step) ---
        print("Training...")
        loss = train_epoch(model, train_loader, optimizer, criterion, hmm_decoder=hmm)
        avg_cer = validate(model, val_loader, hmm)

        print(f"Epoch {epoch} Done. Loss: {loss:.4f} | Avg CER: {avg_cer:.2%}")

        # Save checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()