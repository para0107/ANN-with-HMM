# Offline Handwriting Recognition System (Hybrid ANN-HMM)

This project implements a **Hybrid Artificial Neural Network (ANN) and Hidden Markov Model (HMM)** architecture for offline handwriting recognition.

Unlike modern End-to-End Deep Learning approaches (such as CRNN/CTC) that treat recognition as a black box, this system is built from first principles using **Iterative Expectation-Maximization (EM)** training. It explicitly models the optical probability of character appearance (via ANN) and the temporal elasticity of handwriting structure (via HMM).

---

## 📖 Table of Contents
1. [Theoretical Framework](#-theoretical-framework)
2. [Mathematical Formulation](#-mathematical-formulation)
3. [Technical Architecture](#-technical-architecture)
4. [Experiments & Results](#-experiments--results)
5. [Analysis of Convergence Failure](#-analysis-of-convergence-failure)

---

## 🧠 Theoretical Framework

This system addresses the **Sequence Transduction** problem: converting a variable-length sequence of image frames $X$ into a sequence of characters $W$.

### The Hybrid Hypothesis
Handwriting recognition requires solving two distinct problems simultaneously:
1.  **Optical Recognition:** Identifying what a specific slice of an image looks like (e.g., "This vertical line looks like part of an 'l'").
2.  **Sequence Modeling:** Handling the elasticity of time (e.g., an 'm' might be 10 pixels wide or 20 pixels wide) and enforcing grammar.

We solve this by combining a **Discriminative Model (ANN)** with a **Generative Model (HMM)**.

---

## 📐 Mathematical Formulation

### 1. The Scaled Likelihood
Standard HMMs require the **Emission Probability** $P(x|q)$ (the probability of observing image feature $x$ given state $q$). However, Neural Networks output $P(q|x)$ (the probability of state $q$ given feature $x$).

To bridge this gap, we apply **Bayes' Theorem**. In the log domain, the scaled emission probability is calculated as:

$$\log P(x|q) \approx \underbrace{\log P(q|x)}_{\text{ANN Output}} - \underbrace{\log P(q)}_{\text{State Prior}}$$

This transformation allows the ANN to act as the emission probability estimator for the HMM.

### 2. The Viterbi Algorithm (Forced Alignment)
During the training phase (E-Step), we do not have pixel-level labels (we do not know where character $c$ starts or ends). We use the **Viterbi Algorithm** to find the optimal state sequence $Q^*$ that aligns the image frames $X$ to the ground truth text $W$:

$$Q^* = \underset{Q}{\text{argmax}} \prod_{t=1}^{T} P(x_t | q_t) P(q_t | q_{t-1})$$

### 3. Iterative EM Training
The system learns via the **Expectation-Maximization (EM)** algorithm:
1.  **Flat Start:** Initialize with a heuristic linear alignment.
2.  **E-Step (Alignment):** Fix the ANN weights. Use Viterbi to align the training images to their text labels, generating new "soft" targets.
3.  **M-Step (Update):** Fix the alignment targets. Train the ANN via Backpropagation to maximize the likelihood of these targets. Update HMM transition probabilities via frequency counting.

---

## 🏗 Technical Architecture

### 1. Preprocessing & Feature Extraction
* **Input:** Grayscale line images from the IAM Database.
* **Sliding Window:** A window of width **9 pixels** moves across the image.
* **Feature Vector:** 9 geometrical features (center of gravity, black pixel density, etc.) are extracted per column.
* **Final Input:** A flattened vector of size **540** ($9 \text{ columns} \times 60 \text{ features}$) fed into the ANN.

### 2. Neural Network (Optical Model)
* **Architecture:** Multi-Layer Perceptron (MLP).
* **Structure:** `Input(540) -> Dense(192) -> Dense(128) -> Output(N)`.
* **Activation:** Sigmoid (Hidden layers), LogSoftmax (Output layer).
* **Regularization:** Dropout (0.2).

### 3. Hidden Markov Model (Sequence Model)
* **Topology:** Linear Left-to-Right (Bakis Model).
* **States Per Character:** Configurable (See Experiments).
* **Transitions:** A state $i$ can transition to itself ($i$) or the next state ($i+1$).

---

## 📊 Experiments & Results

### Experiment I: High-Fidelity Topology (Baseline)
The initial experiment was conducted to test the capacity of the model to capture fine-grained character details.

**Configuration:**
* **States Per Character:** 7 (Following classical literature recommendations).
* **Total Output Classes:** 554 ($79 \text{ chars} \times 7 \text{ states} + 1$).
* **Decoding Strategy:** Greedy Decoding.

**Quantitative Results:**

| Metric | Epoch 0 (Start) | Epoch 5 | Epoch 10 (Final) |
| :--- | :--- | :--- | :--- |
| **NLL Loss** | 4.96 | 2.70 | **2.67** |
| **CER (Char Error)** | 97.6% | 132.4% | **166.9%** |
| **WER (Word Error)** | 100% | 213.2% | **253.3%** |

### Analysis of Convergence Failure
Despite the **Loss** decreasing steadily (indicating the ANN was minimizing the negative log-likelihood), the **Error Rates** diverged to >100%.

**The "Stuttering" Phenomenon:**
A Character Error Rate (CER) above 100% indicates massive **Insertion Errors**. The model was transcribing strings significantly longer than the ground truth (e.g., predicting "ttthhheee" instead of "the").

**Root Cause: Topology Mismatch**
1.  **Constraint Violated:** Setting `STATES_PER_CHAR = 7` forces every character to be at least 7 frames wide.
2.  **Reality:** Many characters in the IAM dataset (e.g., 'i', 'l', '1') are only 3-4 frames wide.
3.  **Forced Misalignment:** The Viterbi algorithm was forced to align background noise or neighboring pixels to the character states to satisfy the 7-state length constraint.
4.  **Vicious Cycle:** The ANN learned to classify silence/noise as character features, leading to hallucinated characters during decoding.

---

## 🔮 Future Work & Solution

To resolve the divergence observed in Experiment I, the following changes are proposed for Experiment II:

1.  **Topology Relaxation:** Reduce `STATES_PER_CHAR` from **7** to **3** (Begin, Middle, End). This accommodates narrow characters while still modeling internal structure.
2.  **Smart Decoding:** Update the decoder to only emit characters when a specific **Start State** is entered, rather than on every state change. This will filter out the repetitive "stuttering" artifacts.