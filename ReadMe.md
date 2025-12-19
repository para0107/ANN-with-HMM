# Offline Handwriting Recognition System (Hybrid ANN-HMM)

A comprehensive implementation of an Offline Handwriting Recognition system using a **Hybrid Artificial Neural Network (ANN) and Hidden Markov Model (HMM)** architecture.

This project focuses on transcribing text from the **IAM Handwriting Database**. Unlike modern End-to-End deep learning approaches (like CRNN/CTC), this project builds the system from first principles using **Iterative Expectation-Maximization (EM)** training, making it highly valuable for understanding the foundations of sequence modeling and speech/text recognition.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Technical Architecture](#-technical-architecture)
    - [Preprocessing & Feature Extraction](#1-preprocessing--feature-extraction)
    - [The Hybrid Model (ANN + HMM)](#2-the-hybrid-model-ann--hmm)
    - [The Training Loop (EM Algorithm)](#3-the-training-loop-em-algorithm)
3. [Prerequisites & Hardware](#-prerequisites--hardware)
4. [Installation Guide](#-installation-guide)
5. [Dataset Setup](#-dataset-setup)
6. [Usage Instructions](#-usage-instructions)
7. [Troubleshooting & Performance](#-troubleshooting--performance-notes)

---

## 🔭 Project Overview

This system takes images of handwritten lines as input and produces transcribed text. Because handwriting varies in width and style, we cannot simply map one image column to one character.

Instead, we use a **Hybrid Approach**:
* **The ANN** acts as the "optical model." It looks at a small slice of the image and predicts: *"Which part of which character am I looking at?"*
* **The HMM** acts as the "sequence model." It enforces grammar and structure (e.g., an 'a' must start, then continue, then end). It calculates the most likely path through the ANN's noisy predictions.

---

## 🏗 Technical Architecture

### 1. Preprocessing & Feature Extraction
Before training, raw images are converted into mathematical feature vectors.
* **Deslanting:** Heuristics are applied to correct cursive slant.
* **Sliding Window:** A window of width **9 pixels** moves across the normalized image.
* **Geometrical Features:** For each column in the window, 9 geometrical features are extracted (center of gravity, black pixel density, etc.).
* **Final Vector:** A single frame input to the ANN is a flattened vector of size **540** ($9 \text{ columns} \times 60 \text{ features}$).

### 2. The Hybrid Model (ANN + HMM)
* **Artificial Neural Network (ANN):**
    * **Type:** Multi-Layer Perceptron (MLP).
    * **Input:** 540 dimensions.
    * **Hidden Layers:** 2 layers (192 and 128 neurons) with Sigmoid activation and Dropout (0.2).
    * **Output:** `LogSoftmax` probabilities for every possible HMM state.
    * **Dynamic Sizing:** The output size is calculated dynamically as `(Unique Chars × States Per Char) + 1 Blank`. For the IAM dataset, this is typically **554 neurons**.

* **Hidden Markov Model (HMM):**
    * **Topology:** Linear Left-to-Right (Bakis Model).
    * **States per Character:** **7**. This allows the model to capture the "width" of a character (start, middle, end).
    * **Transition Probability:** Learning how likely it is to stay in the same state (wide character) vs. move to the next (narrow character).

### 3. The Training Loop (EM Algorithm)
We do not have pixel-level labels (we don't know exactly where the 'a' starts in the image). We only have the sentence text. Therefore, we use **Expectation-Maximization**:

1.  **Epoch 0 (Flat Start):** The ANN is trained on a heuristic/flat alignment to initialize weights.
2.  **E-Step (Alignment - CPU):**
    * We run the **Viterbi Algorithm** (Forced Alignment) on every image in the training set.
    * This finds the "best path" matching the known text to the current image features.
    * *Note:* This step is mathematically heavy and runs on the **CPU**.
3.  **M-Step (Update - GPU):**
    * The "best path" from the E-Step becomes the new "Ground Truth" target.
    * The ANN is retrained (Backpropagation) to predict this new path.
    * HMM Priors and Transitions are updated based on statistical counts.

---

## ⚙ Prerequisites & Hardware

**Critical Note on Performance:**
* **GPU:** An NVIDIA GPU (RTX series) is **highly recommended**. Training the ANN on a CPU will take hours per epoch.
* **CPU:** The Alignment phase (Viterbi) is single-threaded CPU work. A fast clock speed is beneficial.
* **RAM:** This project loads feature matrices into memory.
    * **Minimum:** 16 GB System RAM.
    * *Recommendation:* Close web browsers and Electron apps (Teams/Discord) before running to prevent `MemoryError`.

---

## 📦 Installation Guide

### 1. Environment Setup
Create a virtual environment to keep dependencies isolated.
```bash
python -m venv .venv
.venv\Scripts\activate
2. Install PyTorch (CUDA Version)
Do not simply run pip install torch. You must install the version that supports your NVIDIA GPU.

Bash

# For CUDA 12.1 (Recommended for RTX 30xx/40xx)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
3. Install Utilities
Bash

pip install numpy opencv-python matplotlib
🗄 Dataset Setup
The code expects the IAM Handwriting Database in a specific folder structure relative to the src folder.

Create a folder named IAM in the project root.

ASCII Labels: Place lines.txt inside IAM/ascii/.

Images: Extract the line images into IAM/data/lines/.

Final Structure:

Plaintext

ANN-with-HMM/
├── IAM/
│   ├── ascii/lines.txt
│   ├── data/lines/     # Contains 'a01-000u-06.png', etc.
│   └── features/       # Empty initially (Script will fill this)
├── src/
│   ├── main.py
│   ├── preprocess.py
│   ├── model.py
│   ├── hmm.py
│   └── ...
💻 Usage Instructions
Step 1: Feature Extraction
Run the preprocessing script first. This converts images to .npy arrays.

Bash

python src/preprocess.py
Time: 5–15 minutes (Disk dependent).

Output: Populates IAM/features/.

Step 2: Training
Run the main training loop.

Bash

python src/main.py
Epoch 0: Rapid initial training.

Epoch 1+: The EM cycle begins.

The console will say "Aligning training data...".

It will appear stuck here for 10-20 minutes. This is normal (CPU Viterbi processing).

Once alignment finishes, the GPU will kick in for the ANN training pass.

🔧 Troubleshooting & Performance Notes
1. "The console is stuck at 'Aligning training data...'"
Status: Normal Behavior.

Explanation: The script is running the Viterbi algorithm on ~3,500 lines using your CPU. It is not frozen.

Check: Open Task Manager. If Python is using ~15-20% CPU (one full core), it is working. Go grab a coffee.

2. RuntimeError: CUDA error: device-side assert triggered
Status: Critical Error.

Cause: The Dataset generated a label ID (e.g., 553) that is larger than the ANN's output layer (e.g., 546 neurons).

Solution: Ensure main.py dynamically calculates num_classes and passes it to both the ANN and the HMM:

Python

num_classes = (len(CHARS) * STATES_PER_CHAR) + 1
model = ANN(num_classes=num_classes)
hmm = HybridHMM(num_classes=num_classes)
3. ValueError: operands could not be broadcast together
Status: Code mismatch.

Cause: The HMM Priors vector size differs from the ANN output vector size.

Solution: Your hmm.py must be updated to accept num_classes in its __init__ method, ensuring it initializes arrays of size 554 (or whatever the dataset requires), not the hardcoded 546.

4. MemoryError or System Sluggishness
Status: Hardware Bottleneck.

Cause: Loading thousands of numpy arrays into RAM while Chrome/Teams are open.

Solution: Close background applications. Use DataLoader with smaller batch sizes if necessary (though batch size affects the HMM update logic).