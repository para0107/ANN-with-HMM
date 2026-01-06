import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

from dataset import IAMDataset, CHARS, TOTAL_STATES, char_to_state_seq, iam_collate_fn
from model import ANN
from hmm import HybridHMM
from metrics import calculate_cer
from debug_utils import raw_dump

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'IAM', 'features')
XML_DIR = os.path.join(BASE_DIR, 'IAM', 'xml')

BATCH_SIZE = 8
LR = 0.00005
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, optimizer, criterion, hmm_decoder=None):
    model.train()
    total_loss = 0
    batches = 0

    for i, (features, targets, text) in enumerate(dataloader):
        # Features: (Batch, Time, 540)
        # Targets:  (Batch, Time)
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        # Output: (Batch, Time, Classes)
        outputs = model(features)

        # Flatten for NLLLoss: (Batch*Time, Classes) vs (Batch*Time)
        # The loss will ignore entries where target is -1 (padding)
        loss = criterion(outputs.view(-1, TOTAL_STATES), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batches += 1

        if i % 100 == 0:
            # Visualize the FIRST sample in the batch
            with torch.no_grad():
                # Take slice [0] to get (Time, Classes)
                sample_out = outputs[0].cpu().numpy()
                if hmm_decoder:
                    hmm_pred = hmm_decoder.decode(sample_out)
                else:
                    hmm_pred = "?"

            # Raw dump needs (1, Time, Classes) or similar
            raw_pred = raw_dump(outputs[0].unsqueeze(0))

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
            # Validation usually batches too, so we handle it similarly
            features = features.to(DEVICE)
            outputs = model(features)  # (Batch, Time, Classes)

            # Loop over batch elements to decode individually
            for j in range(len(text)):
                # Slice the output for this specific image
                # Note: This includes padding at the end, but HMM decode
                # handles "Space" states well, or we could unpad.
                # For simplicity, we decode the whole padded sequence.
                sample_out = outputs[j].cpu().numpy()
                pred = hmm.decode(sample_out)

                cer = calculate_cer(pred, text[j])
                total_cer += cer
                total_lines += 1

    return total_cer / total_lines if total_lines > 0 else 0


def main():
    print(f"Device: {DEVICE}")
    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # ADDED collate_fn here
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=iam_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=iam_collate_fn)

    print(f"Initializing Model with {TOTAL_STATES} Total States.")
    model = ANN(num_classes=TOTAL_STATES).to(DEVICE)

    # Initialize HMM with stricter transitions (Self-Loop 0.9)
    hmm = HybridHMM(num_classes=TOTAL_STATES)
    hmm.transitions[:, 0] = 0.9  # Encourage staying
    hmm.transitions[:, 1] = 0.1  # Discourage skipping

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ADDED ignore_index=-1 to skip padded areas
    criterion = nn.NLLLoss(ignore_index=-1)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch} ---")

        if epoch > 1:
            print("Aligning (Viterbi)...")
            model.eval()
            hmm.reset_accumulators()

            # Alignment is tricky with batches/padding.
            # It's safer/easier to iterate the dataset one by one for alignment
            # because we need to update the specific index in the dataset.
            for i in range(len(train_subset)):
                real_idx = train_subset.indices[i]
                feat, _, text = full_dataset.get_item_with_text(real_idx)
                feat = feat.to(DEVICE)  # (Time, 540)

                with torch.no_grad():
                    # unsqueeze to make it (1, Time, 540) for model
                    out = model(feat.unsqueeze(0)).squeeze(0).cpu().numpy()

                scaled = hmm.get_scaled_emissions(out)

                state_seq = []
                for char in text:
                    state_seq.extend(char_to_state_seq(char))

                if len(state_seq) > 0:
                    path = hmm.forced_alignment(scaled, state_seq)
                    if path is not None:
                        full_dataset.update_target_at_index(real_idx, torch.from_numpy(path).long())

            hmm.update_parameters()
        else:
            print("Skipping Alignment (Warmup Phase: Standard Flat Start)")

        print("Training...")
        loss = train_epoch(model, train_loader, optimizer, criterion, hmm_decoder=hmm)
        avg_cer = validate(model, val_loader, hmm)

        print(f"Epoch {epoch} Done. Loss: {loss:.4f} | Avg CER: {avg_cer:.2%}")
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()