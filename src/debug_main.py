import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os


from dataset import IAMDataset, CHARS, TOTAL_STATES
from model import ANN
from hmm import HybridHMM
from debug_utils import raw_dump


LR = 0.001
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, 'IAM', 'features')
XML_DIR = os.path.join(BASE_DIR, 'IAM', 'xml')


def main():
    print(f"--- DEBUG MODE: Overfitting One Image ---")
    print(f"Using Device: {DEVICE}")

    full_dataset = IAMDataset(FEATURE_DIR, XML_DIR)

    if len(full_dataset) == 0:
        print("Error: No data found.")
        return

    debug_subset = Subset(full_dataset, [0])
    train_loader = DataLoader(debug_subset, batch_size=1, shuffle=False)

    _, _, sample_text = full_dataset[0]
    print(f"Target Text to Learn: '{sample_text}'")


    num_classes = TOTAL_STATES
    print(f"Initializing for {num_classes} dynamic states.")

    model = ANN(num_classes=num_classes).to(DEVICE)
    hmm = HybridHMM(num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    model.train()

    print("\nStarting Training (Expect Loss to drop to near 0)...")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0

        for features, targets, text in train_loader:
            features = features.to(DEVICE).squeeze(0)
            targets = targets.to(DEVICE).squeeze(0)

            optimizer.zero_grad()
            outputs = model(features)



            loss = criterion(outputs, targets)
            loss.backward()

            total_grad = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad += p.grad.abs().mean().item()
            print(f"Avg Gradient Magnitude: {total_grad:.6f}")


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)



            optimizer.step()
            total_loss += loss.item()

            if epoch % 10 == 0:
                with torch.no_grad():
                    raw_str = raw_dump(outputs.unsqueeze(0))
                    hmm_str = hmm.decode(outputs.cpu().numpy())

                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
                print(f"  Raw: '{raw_str}'")
                print(f"  HMM: '{hmm_str}'")

    print("\n--- Debug Finished ---")
    print("If 'HMM' matches 'Target Text', the model works.")


if __name__ == "__main__":
    main()