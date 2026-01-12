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
BASE_LR = 0.005
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_EPOCHS = 5


def get_lr(epoch, base_lr=0.005, warmup_epochs=3):
    """Learning rate warmup then constant."""
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    return base_lr


def train_epoch(model, dataloader, optimizer, criterion, hmm_decoder=None, epoch=1):
    model.train()
    total_loss = 0
    batches = 0
    non_empty_predictions = 0
    total_grad = 0

    for i, (features, targets, text) in enumerate(dataloader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs.view(-1, TOTAL_STATES), targets.view(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [Batch {i}] Skipping - NaN/Inf loss")
            continue

        loss.backward()

        batch_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                batch_grad += p.grad.abs().mean().item()
        total_grad += batch_grad

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        batches += 1

        with torch.no_grad():
            sample_out = outputs[0].cpu().numpy()
            if hmm_decoder:
                pred = hmm_decoder.decode(sample_out)
                if len(pred.strip()) > 0:
                    non_empty_predictions += 1

        if i % 100 == 0:
            with torch.no_grad():
                sample_out = outputs[0].cpu().numpy()
                if hmm_decoder:
                    hmm_pred = hmm_decoder.decode(sample_out)
                else:
                    hmm_pred = "?"

            raw_pred = raw_dump(outputs[0].unsqueeze(0))

            print(f"  [Batch {i}] Truth: '{text[0]}'")
            print(f"             Raw:   '{raw_pred}'")
            print(f"             HMM:   '{hmm_pred}'")
            print("-" * 20)

    avg_loss = total_loss / batches if batches > 0 else 0
    non_empty_rate = non_empty_predictions / batches if batches > 0 else 0
    avg_grad = total_grad / batches if batches > 0 else 0

    print(f"Avg Gradient Magnitude: {avg_grad:.6f}")

    return avg_loss, non_empty_rate


def validate(model, dataloader, hmm):
    model.eval()
    total_cer = 0
    total_lines = 0

    with torch.no_grad():
        for features, _, text in dataloader:
            features = features.to(DEVICE)
            outputs = model(features)

            for j in range(len(text)):
                sample_out = outputs[j].cpu().numpy()
                pred = hmm.decode(sample_out)

                cer = calculate_cer(pred, text[j])
                total_cer += cer
                total_lines += 1

    return total_cer / total_lines if total_lines > 0 else 1.0


def align_dataset(model, hmm, full_dataset, train_subset):
    model.eval()
    hmm.reset_accumulators()

    successful_alignments = 0
    total_samples = len(train_subset)

    for i in range(total_samples):
        real_idx = train_subset.indices[i]
        feat, _, text = full_dataset.get_item_with_text(real_idx)

        if len(text) == 0:
            continue

        feat = feat.to(DEVICE)

        with torch.no_grad():
            out = model(feat.unsqueeze(0)).squeeze(0).cpu().numpy()

        scaled = hmm.get_scaled_emissions(out, penalize_space=False)
        path = hmm.forced_alignment(scaled, text)

        if path is not None:
            full_dataset.update_target_at_index(real_idx, torch.from_numpy(path).long())
            successful_alignments += 1

    hmm.update_parameters(smoothing=1.0)

    return successful_alignments, total_samples


def main():
    print(f"Device: {DEVICE}")
    print(f"Total HMM States: {TOTAL_STATES}")
    print(f"Warmup Epochs (no re-alignment): {WARMUP_EPOCHS}")
    print(f"Base Learning Rate: {BASE_LR}")

    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)
    print(f"Dataset size: {len(full_dataset)}")

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=iam_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=iam_collate_fn)

    print(f"Initializing Model with {TOTAL_STATES} Total States.")
    model = ANN(num_classes=TOTAL_STATES).to(DEVICE)

    hmm = HybridHMM(num_classes=TOTAL_STATES, min_state_duration=1)

    optimizer = optim.Adam(model.parameters(), lr=BASE_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion = nn.NLLLoss(ignore_index=-1)

    best_cer = 1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 50}")
        print(f"--- Epoch {epoch} ---")
        print(f"{'=' * 50}")

        current_lr = get_lr(epoch, base_lr=BASE_LR, warmup_epochs=3)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f"Learning Rate: {current_lr:.6f}")

        if epoch > WARMUP_EPOCHS:
            print(f"Aligning with Viterbi...")
            successful, total = align_dataset(model, hmm, full_dataset, train_subset)
            print(f"Alignment: {successful}/{total} successful")
        else:
            print(f"Warmup Phase: Using flat-start targets (epoch {epoch}/{WARMUP_EPOCHS})")

        print("Training...")
        loss, non_empty_rate = train_epoch(model, train_loader, optimizer, criterion, hmm_decoder=hmm, epoch=epoch)

        print(f"\nValidating...")
        avg_cer = validate(model, val_loader, hmm)

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Loss: {loss:.4f}")
        print(f"Non-empty prediction rate: {non_empty_rate:.2%}")
        print(f"Avg CER: {avg_cer:.2%}")

        scheduler.step(loss)

        if avg_cer < best_cer:
            best_cer = avg_cer
            torch.save(model.state_dict(), "model_best.pth")
            print(f"New best model saved! CER: {best_cer:.2%}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

        if epoch > WARMUP_EPOCHS and non_empty_rate < 0.1:
            print("\nWARNING: Model may be collapsing (>90% empty predictions)")
            print("Consider reducing learning rate or checking data.")


if __name__ == "__main__":
    main()
