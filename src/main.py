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
WEIGHTS_DIR = os.path.join(BASE_DIR, 'src', 'weights')

BATCH_SIZE = 8
BASE_LR = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_EPOCHS = 2
NUM_WORKERS = 4
PIN_MEMORY = True


def get_lr(epoch, base_lr=0.001, warmup_epochs=2):
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
                counts[int(t)] += 1

    weights = np.ones(num_classes)
    total = counts.sum()
    if total > 0:
        for i in range(num_classes):
            if counts[i] > 0:
                weights[i] = total / (num_classes * counts[i])

    weights = np.clip(weights, 0.1, 10.0)
    print(f"Weight range: {weights.min():.3f} - {weights.max():.3f}")
    return torch.FloatTensor(weights)


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

        if batch_idx % 100 == 0:
            with torch.no_grad():
                raw_str = raw_dump(outputs[0:1])
                hmm_str = hmm_decoder.decode(outputs[0].cpu().numpy()) if hmm_decoder else ""
                ground_truth = texts[0]  # First sample's ground truth
            print(f"  Batch {batch_idx}: Loss={loss.item():.4f}")
            print(f"    GT:  '{ground_truth}'")
            print(f"    Raw: '{raw_str}'")
            print(f"    HMM: '{hmm_str}'")

    avg_loss = total_loss / len(dataloader)
    return avg_loss, 0.0


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
    print(f"Total HMM states: {TOTAL_STATES}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    dataset = IAMDataset(FEATURE_DIR, XML_DIR)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("No data found!")
        return

    # Split: 80% train, 10% validation, 10% test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=iam_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=iam_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=iam_collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = ANN(num_classes=TOTAL_STATES).to(DEVICE)
    hmm = HybridHMM(num_classes=TOTAL_STATES)

    weights = compute_class_weights(dataset, TOTAL_STATES).to(DEVICE)
    criterion = nn.NLLLoss(weight=weights, ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR)

    best_cer = float('inf')

    for epoch in range(1, EPOCHS + 1):
        lr = get_lr(epoch, BASE_LR, WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f"\n=== Epoch {epoch}/{EPOCHS} (LR: {lr:.6f}) ===")

        train_loss, _ = train_epoch(model, train_loader, optimizer, criterion, hmm, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        val_cer = validate(model, val_loader, hmm)
        print(f"Validation CER: {val_cer:.4f}")

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pth"))
            print(f"  -> New best model saved!")

        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"model_epoch_{epoch}.pth"))

    # Final evaluation on test set
    print("\n=== Final Test Evaluation ===")
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "best_model.pth")))
    test_cer = validate(model, test_loader, hmm)
    print(f"Test CER: {test_cer:.4f}")

    print(f"\nTraining complete. Best Val CER: {best_cer:.4f}, Test CER: {test_cer:.4f}")


if __name__ == "__main__":
    main()
