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
BASE_LR = 0.001
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_EPOCHS = 3


def get_lr(epoch, base_lr=0.001, warmup_epochs=3):
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    return base_lr


def compute_class_weights(dataset, num_classes):
    """Compute inverse frequency weights for each class."""
    print("Computing class weights...")
    counts = np.zeros(num_classes)

    sample_size = min(len(dataset), 2000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    for i in indices:
        _, targets, _ = dataset[i]
        for t in targets.numpy():
            if 0 <= t < num_classes:
                counts[t] += 1

    weights = np.ones(num_classes)
    total = counts.sum()
    if total > 0:
        for i in range(num_classes):
            if counts[i] > 0:
                weights[i] = total / (num_classes * counts[i])
            else:
                weights[i] = 1.0

    weights = np.clip(weights, 0.1, 10.0)
    print(f"Weight range: {weights.min():.3f} - {weights.max():.3f}")
    return torch.FloatTensor(weights)


def train_epoch(model, dataloader, optimizer, criterion, hmm_decoder=None, epoch=1):
    model.train()
    total_loss = 0
    total_cer = 0
    num_samples = 0
    non_empty = 0

    for batch_idx, (features, targets, texts) in enumerate(dataloader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(features)

        batch_size, time_steps, num_classes = outputs.size()
        outputs_flat = outputs.view(-1, num_classes)
        targets_flat = targets.view(-1)

        loss = criterion(outputs_flat, targets_flat)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss at batch {batch_idx}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            with torch.no_grad():
                raw_str = raw_dump(outputs[0:1])
                if hmm_decoder:
                    hmm_str = hmm_decoder.decode(outputs[0].cpu().numpy())
                else:
                    hmm_str = raw_str

                truth = texts[0]
                cer = calculate_cer(hmm_str, truth)

                if len(hmm_str.strip()) > 0:
                    non_empty += 1

                unique_chars = len(set(hmm_str))
                collapse_warning = " [COLLAPSE]" if unique_chars < 5 else ""

                print(f"  [Batch {batch_idx}] Truth: '{truth}'")
                print(f"             Raw:   '{raw_str}'")
                print(f"             HMM:   '{hmm_str}'{collapse_warning}")
                print("-" * 20)

        for i in range(len(texts)):
            if hmm_decoder:
                pred = hmm_decoder.decode(outputs[i].cpu().detach().numpy())
            else:
                pred = raw_dump(outputs[i:i + 1])
            total_cer += calculate_cer(pred, texts[i])
            num_samples += 1
            if len(pred.strip()) > 0:
                non_empty += 1

    avg_loss = total_loss / len(dataloader)
    avg_cer = total_cer / max(num_samples, 1)
    non_empty_rate = non_empty / max(num_samples, 1) * 100

    print(f"\nEpoch {epoch} Summary ---")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Non-empty prediction rate: {non_empty_rate:.2f}%")
    print(f"Avg CER: {avg_cer * 100:.2f}%")

    return avg_loss, avg_cer


def validate(model, dataloader, hmm):
    model.eval()
    total_cer = 0
    num_samples = 0

    with torch.no_grad():
        for features, targets, texts in dataloader:
            features = features.to(DEVICE)
            outputs = model(features)

            for i in range(len(texts)):
                pred = hmm.decode(outputs[i].cpu().numpy())
                total_cer += calculate_cer(pred, texts[i])
                num_samples += 1

    return total_cer / max(num_samples, 1)


def align_dataset(model, hmm, full_dataset, train_subset):
    model.eval()
    hmm.reset_accumulators()

    print("Re-aligning dataset with forced alignment...")
    aligned_count = 0

    with torch.no_grad():
        for idx in train_subset.indices:
            windows, _, text = full_dataset.get_item_with_text(idx)
            if windows is None or len(text) == 0:
                continue

            windows = windows.to(DEVICE)
            windows = (windows - windows.mean()) / (windows.std() + 1e-6)

            outputs = model(windows)
            log_probs = outputs.cpu().numpy()

            scaled_emissions = hmm.get_scaled_emissions(log_probs)
            new_path = hmm.forced_alignment(scaled_emissions, text)

            if new_path is not None:
                full_dataset.update_target_at_index(idx, torch.from_numpy(new_path).long())
                aligned_count += 1

    hmm.update_parameters()
    print(f"Aligned {aligned_count} samples")
    return aligned_count


def main():
    print(f"Device: {DEVICE}")
    print(f"Total HMM States: {TOTAL_STATES}")
    print(f"Warmup Epochs (no re-alignment): {WARMUP_EPOCHS}")
    print(f"Base Learning Rate: {BASE_LR}")

    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)
    print(f"Dataset size: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("Error: No data found.")
        return

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=iam_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=iam_collate_fn)

    print(f"Initializing Model with {TOTAL_STATES} Total States.")
    model = ANN(num_classes=TOTAL_STATES).to(DEVICE)

    hmm = HybridHMM(num_classes=TOTAL_STATES, min_state_duration=1)

    class_weights = compute_class_weights(full_dataset, TOTAL_STATES).to(DEVICE)
    criterion = nn.NLLLoss(weight=class_weights, ignore_index=-1)

    optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_cer = 1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 50}")
        print(f"--- Epoch {epoch} ---")
        print(f"{'=' * 50}")

        current_lr = get_lr(epoch, base_lr=BASE_LR, warmup_epochs=WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f"Learning Rate: {current_lr:.6f}")

        if epoch <= WARMUP_EPOCHS:
            print(f"Warmup Phase: Using flat-start targets (epoch {epoch}/{WARMUP_EPOCHS})")
        else:
            if epoch == WARMUP_EPOCHS + 1:
                full_dataset.clear_target_cache()
            aligned = align_dataset(model, hmm, full_dataset, train_subset)
            print(f"Re-alignment Phase: Updated {aligned} targets")

        print("Training...")
        train_loss, train_cer = train_epoch(model, train_loader, optimizer, criterion, hmm, epoch)

        print("Validating...")
        val_cer = validate(model, val_loader, hmm)
        print(f"Validation CER: {val_cer * 100:.2f}%")

        scheduler.step(val_cer)

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), f"model_best.pth")
            print(f"New best model saved! CER: {best_cer * 100:.2f}%")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
            print(f"Checkpoint saved: model_epoch_{epoch}.pth")

    print(f"\nTraining Complete. Best CER: {best_cer * 100:.2f}%")


if __name__ == "__main__":
    main()
