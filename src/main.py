import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import os
import copy

from dataset import IAMDataset, CHARS, TOTAL_STATES, iam_collate_fn
from model import ANN
from hmm import HybridHMM
from metrics import calculate_cer
from debug_utils import raw_dump

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'IAM', 'features')
XML_DIR = os.path.join(BASE_DIR, 'IAM', 'xml')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'src', 'weights')

BATCH_SIZE = 8
BASE_LR = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_EPOCHS = 5  # UPDATED: Increased to 5 to prevent model collapse
NUM_WORKERS = 4
PIN_MEMORY = True


def get_lr(epoch, base_lr=0.001, warmup_epochs=2):
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    return base_lr


def compute_class_weights(dataset, num_classes):
    print("Computing class weights...")
    counts = np.zeros(num_classes)
    sample_size = min(len(dataset), 2000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)

    for i in indices:
        _, targets, _ = dataset[i]
        for t in targets.numpy():
            if 0 <= t < num_classes:
                counts[int(t)] += 1

    weights = np.ones(num_classes)
    total = counts.sum()
    if total > 0:
        for i in range(num_classes):
            if counts[i] > 0:
                weights[i] = total / (num_classes * counts[i])

    weights = np.clip(weights, 0.1, 10.0)
    return torch.FloatTensor(weights)


def align_dataset(model, subset, hmm):
    """
    Viterbi Training: Runs the model on the training set to find the
    optimal state sequence (alignment) for the ground truth text.
    Updates the dataset cache with these refined targets.
    """
    print(f"--- Re-aligning Training Data (Forced Alignment) ---")
    model.eval()

    # Unwrap Subset to get to the actual dataset and original indices
    base_dataset = subset.dataset if isinstance(subset, Subset) else subset
    indices = subset.indices if isinstance(subset, Subset) else range(len(subset))

    update_count = 0

    # Reset HMM counters for re-estimation
    hmm.reset_accumulators()

    with torch.no_grad():
        for i, original_idx in enumerate(indices):
            # 1. Get data (bypass collate to get raw size)
            features, _, text = base_dataset.get_item_with_text(original_idx)
            features = features.unsqueeze(0).to(DEVICE)  # (1, T, F)

            # 2. Network forward
            outputs = model(features)
            log_probs = outputs.squeeze(0).cpu().numpy()  # (T, C)

            # 3. Scaled Emissions for HMM
            scaled_emissions = hmm.get_scaled_emissions(log_probs)

            # 4. Forced Alignment (Match text to image frames)
            new_path = hmm.forced_alignment(scaled_emissions, text)

            if new_path is not None:
                # 5. Update Dataset Cache
                base_dataset.update_target_at_index(original_idx, torch.from_numpy(new_path).long())
                update_count += 1

            if i % 2000 == 0 and i > 0:
                print(f"   Aligned {i} samples...")

    # UPDATED: Commented out to prevent HMM from learning bad transitions early on
    # hmm.update_parameters()
    print(f"--- Alignment Complete: Updated {update_count}/{len(subset)} samples. (HMM Params Frozen) ---")


def train_epoch(model, dataloader, optimizer, criterion, hmm_decoder=None, epoch=1):
    model.train()
    total_loss = 0

    for batch_idx, (features, targets, texts) in enumerate(dataloader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(features)

        B, T, C = outputs.shape
        outputs_flat = outputs.view(-1, C)
        targets_flat = targets.view(-1)

        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 200 == 0:
            with torch.no_grad():
                raw_str = raw_dump(outputs[0:1])
                hmm_str = hmm_decoder.decode(outputs[0].cpu().numpy()) if hmm_decoder else ""
                ground_truth = texts[0]
            print(f"  Batch {batch_idx}: Loss={loss.item():.4f}")
            print(f"    GT : '{ground_truth}'")
            print(f"    Raw: '{raw_str[:100]}...'")
            print(f"    HMM: '{hmm_str[:100]}...'")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, hmm):
    model.eval()
    total_cer = 0
    num_samples = 0

    with torch.no_grad():
        for features, targets, texts in dataloader:
            features = features.to(DEVICE)
            outputs = model(features)

            for i in range(len(texts)):
                pred_text = hmm.decode_greedy(outputs[i].cpu().numpy())
                cer = calculate_cer(pred_text, texts[i])
                total_cer += cer
                num_samples += 1

    avg_cer = total_cer / max(num_samples, 1)
    return avg_cer


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    dataset = IAMDataset(FEATURE_DIR, XML_DIR)
    if len(dataset) == 0:
        print("No data found!")
        return

    # Split: 90% train, 5% val, 5% test
    train_size = int(0.90 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=iam_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=iam_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = ANN(num_classes=TOTAL_STATES).to(DEVICE)
    hmm = HybridHMM(num_classes=TOTAL_STATES)

    weights = compute_class_weights(dataset, TOTAL_STATES).to(DEVICE)
    criterion = nn.NLLLoss(weight=weights, ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR)

    best_cer = float('inf')

    for epoch in range(1, EPOCHS + 1):
        lr = get_lr(epoch, BASE_LR, 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f"\n=== Epoch {epoch}/{EPOCHS} (LR: {lr:.6f}) ===")

        # 1. Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, hmm, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        # 2. Validate
        val_cer = validate(model, val_loader, hmm)
        print(f"Validation CER: {val_cer:.4f}")

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pth"))
            print(f"  -> New best model saved!")

        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"model_epoch_{epoch}.pth"))

        # 3. ALIGNMENT STEP
        if epoch >= WARMUP_EPOCHS:
            align_dataset(model, train_dataset, hmm)

    print(f"\nTraining complete. Best Val CER: {best_cer:.4f}")


if __name__ == "__main__":
    main()