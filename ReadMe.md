# Offline Handwriting Recognition System (Hybrid ANN-HMM)

## 1. Project Overview
This project implements a **Hybrid Artificial Neural Network (ANN) and Hidden Markov Model (HMM)** architecture for unconstrained offline handwriting recognition. Unlike modern End-to-End Deep Learning approaches (such as CRNN/CTC) that often treat recognition as a "black box," this system is constructed from first principles using **Iterative Expectation-Maximization (EM)** training.

The core philosophy is to explicitly model the two distinct sub-problems of handwriting recognition:
1.  **Optical Probability (The Eye):** Identifying the visual characteristics of a specific image frame (handled by the ANN).
2.  **Sequential Grammar (The Brain):** Modeling the elasticity of time (duration of characters) and the valid progression of states (handled by the HMM).

---

## 2. Theoretical Framework & Mathematical Formulation

The system addresses the **Sequence Transduction** problem: converting a variable-length sequence of image feature vectors $X = (x_1, x_2, ..., x_T)$ into a sequence of characters $W$.

### The Hybrid Hypothesis
In a standard HMM, we model the joint probability $P(X, W)$. However, HMMs typically use Gaussian Mixture Models (GMMs) to model the observation probability $P(x_t | q_k)$, which struggle with high-dimensional image data.

We replace the GMM with a **Neural Network (ANN)**. The ANN estimates the posterior probability $P(q_k | x_t)$ (the probability of being in state $k$ given the image frame $x_t$). Using Bayes' Rule, we convert this to a "scaled likelihood" suitable for the HMM decoding:

$$\frac{P(x_t | q_k)}{P(x_t)} = \frac{P(q_k | x_t)}{P(q_k)}$$

Where:
* $P(q_k | x_t)$ is the output of the Neural Network.
* $P(q_k)$ is the prior probability of state $k$ (estimated from training data frequency).

### Expectation-Maximization (EM) Training
The system creates a "virtuous cycle" of training without requiring frame-level labels (i.e., we know the image says "The", but we don't know which pixel is 'T').

1.  **E-Step (Forced Alignment):** The HMM uses the current ANN to find the most likely alignment (path) between the image frames and the ground truth text. This assigns a specific state label to every frame.
2.  **M-Step (Training):** The ANN is trained via Backpropagation to predict these new labels.
3.  **Repeat:** As the ANN gets better, the alignment improves. As the alignment improves, the ANN gets better training data.

---

## 3. Technical Architecture & Implementation

### A. Preprocessing Pipeline (`preprocess.py`)
Raw handwriting images are highly variable. We implement a rigorous geometric normalization pipeline to reduce variance before feature extraction.

1.  **Binarization & Noise Removal:** Images are converted to grayscale, inverted (ink=white), and thresholded using Otsu’s method to separate ink from background.
2.  **Slope Correction:** A linear regression is performed on the ink pixel coordinates to detect the general slope of the line. The image is rotated via affine transformation to align the text baseline horizontally.
3.  **Slant Correction:** Second-order moments are calculated to estimate the dominant shear angle of the writing (italic tilt). A shear transformation makes the vertical strokes upright.
4.  **Grid-Based Feature Extraction:** A sliding window moves across the image. Each window is divided into a $20 \times 1$ grid. For each cell, we extract 3 features:
    * **Normalized Gray Level:** Ink density.
    * **Horizontal Derivative:** Rate of change in ink intensity (X-axis).
    * **Vertical Derivative:** Rate of change in ink intensity (Y-axis).
    * *Total Input Dimension:* $20 \text{ cells} \times 3 \text{ features} \times 9 \text{ context frames} = 540 \text{ inputs}$.

### B. Dynamic HMM Topology (`dataset.py` & `hmm.py`)
Instead of a rigid topology, we implemented a **Dynamic State Allocation** system to reflect the physical reality of handwriting.

* **Variable Character Length:** Wide characters (e.g., 'm', 'w') are assigned more states (5-9) than narrow characters (e.g., 'i', 'l', which get 2-3 states). This prevents the "Time Distortion" problem where narrow letters are forced to stretch unnaturally.
* **The "Silence" Model:** The Space character (`' '`) is modeled with a single state ($N=1$) with high self-loop probability. This allows the model to handle variable inter-word gaps without hallucinating "ghost characters" in the empty space.

### C. Neural Network (`model.py`)
We modernized the architecture proposed in classical literature (which relied on Sigmoid/Tanh) to ensure stable convergence.

* **Architecture:** Multilayer Perceptron (MLP).
* **Activations:** **ReLU** (Rectified Linear Unit) prevents the vanishing gradient problem.
* **Regularization:** **Batch Normalization** ensures input stability at each layer, and **Dropout** prevents overfitting.
* **Output:** Log-Softmax over the total number of HMM states (approx. 200-300 states depending on topology).

---

## 4. Experiments & Results

We conducted a series of iterative experiments to stabilize the training process.

### Experiment I: The "Stuttering" Divergence
* **Setup:** Fixed topology (7 states per character), Sigmoid activations.
* **Observation:** Loss decreased, but Character Error Rate (CER) exploded to >100%.
* **Analysis:** The fixed 7-state constraint forced narrow letters (like 'i') to align with background noise to fill the required duration. The model learned to interpret noise as character features, leading to repetitive insertions (e.g., "ttthhheee").

### Experiment II: The "Space Collapse"
* **Setup:** Modernized ANN (ReLU), Flat Start with Padding (adding spaces to image edges).
* **Observation:** The model converged to a local minimum where it predicted *only* spaces for the entire sequence.
* **Analysis:** By padding the targets with spaces, we inadvertently biased the model. Since 90% of a handwritten image is background, the model learned that predicting "Space" is the safest way to minimize loss globally.

### Experiment III: Dynamic Topology & Aggressive Alignment
* **Setup:** * **Dynamic States:** 'm'=5, 'i'=2, 'Space'=1.
    * **No Warmup:** Switched immediately to Viterbi alignment (skipping the misleading "Flat Start" phase).
* **Result:** **Success.** The model successfully began transcribing distinct characters. The "Silence" state successfully absorbed inter-word gaps, and the dynamic topology allowed distinct modeling of wide vs. narrow characters.

---

## 5. Comparison with State-of-the-Art (SOTA)

Compared to the seminal paper *"Improving Offline Handwritten Text Recognition with Hybrid HMM/ANN Models"* (España-Boquera et al.), our implementation features several modernizations and simplifications:

| Component | Original Paper Approach | Our Project Implementation | Impact |
| :--- | :--- | :--- | :--- |
| **Slope Correction** | **Baseline Detection:** Tracks the lower contour of the letter body, ignoring descenders (tails of 'g', 'y'). | **Linear Regression:** Fits a line to all ink pixels. | Our method is simpler but may be less robust to words with many descenders. |
| **ANN Activation** | **Sigmoid/Tanh:** Standard for 2011, but prone to vanishing gradients. | **ReLU + BatchNorm:** Modern Deep Learning standard. | **Improved Stability:** Faster convergence and ability to train deeper networks. |
| **Decoding** | **Constrained (Language Model):** Uses a dictionary/N-gram model to force output into valid words. | **Greedy / Unconstrained:** Outputs the raw sequence of most likely characters. | **Trade-off:** Our raw character accuracy is high, but we lack the "spell-check" layer to fix ambiguity (e.g., "thue" vs "the"). |
| **Features** | **Geometric Moments:** Center of gravity, second-order moments (Marti & Bunke). | **Derivative Features:** Grid-based gray level and gradients. | Computationally lighter, though potentially less discriminative for complex shapes. |

---

## 6. Future Work

To bridge the gap between the current prototype and human-level performance, the following steps are recommended:

1.  **Language Model Integration:** Implement a Token-Passing algorithm or Weighted Finite State Transducer (WFST) to constrain decoding to a valid English lexicon.
2.  **Robust Baseline Detection:** Upgrade the preprocessing pipeline to detect the "core" of the text line, ignoring ascenders/descenders for more accurate rotation.
3.  **Data Augmentation:** Introduce random shears, rotations, and erosions during training to make the ANN robust to varied handwriting styles.