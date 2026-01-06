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

## 7. Recent Experimental Iterations & Troubleshooting

The development of this Hybrid ANN-HMM system required solving several critical convergence pathologies common in sequence modeling. Below is the documentation of the specific failures encountered and the architectural changes made to resolve them.

### Experiment IV: The "Insertion Explosion" (Stuttering Pathology)
* **The Symptom:** During early training, the model predicted character sequences that were nearly as long as the input image width. For an image containing "The", the output would resemble `TTTThhhheeeee...` (hundreds of characters long).
* **The Diagnosis:** **Transition Probability Mismatch.**
    * The HMM transition matrix was initialized with a self-loop probability $P(q_i | q_i) = 0.5$ and a next-state probability $P(q_{i+1} | q_i) = 0.5$.
    * This implies an expected state duration of only 2 frames. However, in the IAM dataset at 300 DPI, a single character often spans 20–30 frames.
    * **Result:** The HMM was statistically forced to exit the state too early. To consume the remaining frames of the character ink, it had no choice but to re-enter the character or hallucinate new ones, causing massive insertion errors.
* **The Fix:** We modified the HMM initialization to enforce **"Sticky States"**. The self-loop probability was increased to **0.9** (encouraging duration), and the exit probability reduced to **0.1**.
* **Outcome:** The output length stabilized to match the ground truth length, resolving the stuttering.

### Experiment V: Batch Processing & Tensor Dimensionality
* **The Problem:** Training with `BATCH_SIZE = 1` resulted in noisy gradient updates and slow convergence. Increasing the batch size caused `RuntimeError` due to variable image widths in the IAM dataset.
* **The Technical Challenge:** Standard Neural Networks expect fixed-size tensors, but handwriting lines vary in length $T$.
    * Attempting to simply flatten the input resulted in `Mat1 and Mat2 shape mismatch` errors because the temporal dimension $T$ was being merged with the feature dimension $F$ incorrectly.
* **The Fix:**
    1.  **Custom Collation:** We implemented a `collate_fn` that pads all feature sequences in a batch to the length of the longest sequence (using zero-padding).
    2.  **3D Tensor Handling:** We refactored the ANN's `forward` pass to accept `(Batch, Time, Features)`. It temporarily merges `Batch` and `Time` to process frames independently ($B \cdot T, F$) and then reconstructs the sequence structure before passing it to the HMM.
* **Outcome:** Enabled stable batch training (Batch Size 8), significantly smoothing the loss curve.

### Experiment VI: "Margin Poisoning" & The Space Collapse
* **The Symptom:** After the HMM alignment phase began, the model stopped predicting characters entirely and converged to predicting **only Spaces** (a blank string), despite the Loss decreasing.
* **The Diagnosis:** **Toxic Flat Start Alignment.**
    * In the "Warmup Phase," we used a linear Flat Start that stretched the target text (e.g., "The cat") across the *entire* width of the image (0 to $W$).
    * However, handwriting images typically have wide white margins on the left and right.
    * **The Poison:** The model was forced to align the first letter 'T' with the white pixels of the left margin. The Neural Network learned the incorrect association: $\text{White Background} \approx \text{Character Ink}$.
    * Since 80% of an image is background, the model optimized the loss by predicting the "Space" state everywhere.
* **The Fix:** **Centered Flat Start.**
    * We modified the alignment logic in `dataset.py`. Instead of stretching the text, we now estimate the physical length of the text (based on average character width) and **center** it in the target array.
    * The remaining frames at the start and end are explicitly assigned to the **Space State**.
* **Outcome:** This teaches the model that margins are silence, protecting the character states from learning background noise. This is currently the active experiment, aiming to allow characters to re-emerge from the correct positions.