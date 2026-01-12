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

        batch_size, time_steps, num_classes = outputs.size()
        outputs_flat = outputs.view(-1, num_classes)
        targets_flat = targets.view(-1)

        loss = criterion(outputs_flat, targets_flat)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        # Only decode/print every 100 batches
        if batch_idx % 100 == 0:
            with torch.no_grad():
                raw_str = raw_dump(outputs[0:1])
                hmm_str = hmm_decoder.decode_greedy(outputs[0].cpu().numpy()) if hmm_decoder else raw_str
                print(f"  [Batch {batch_idx}] Truth: '{texts[0]}'")
                print(f"             Raw:   '{raw_str}'")
                print(f"             HMM:   '{hmm_str}'")
                print("-" * 20)

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
                pred = hmm.decode(outputs[i].cpu().numpy())
                total_cer += calculate_cer(pred, texts[i])
                num_samples += 1

    avg_cer = total_cer / max(num_samples, 1)
    return avg_cer


def main():
    print(f"Using device: {DEVICE}")
    print(f"Total HMM states: {TOTAL_STATES}")

    dataset = IAMDataset(FEATURE_DIR, XML_DIR)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("No data found!")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=iam_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=iam_collate_fn)

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

        if epoch % 5 == 0:
            val_cer = validate(model, val_loader, hmm)
            print(f"Validation CER: {val_cer:.4f}")

            if val_cer < best_cer:
                best_cer = val_cer
                torch.save(model.state_dict(), f"model_best.pth")
                print(f"New best model saved! CER: {best_cer:.4f}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    print(f"\nTraining complete. Best CER: {best_cer:.4f}")


if __name__ == "__main__":
    main()
