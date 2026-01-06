## 1. Slope Correction (Preprocessing)

**The Challenge:** Handwritten lines often drift upwards or downwards. If not corrected, a sliding window will capture different parts of a character depending on its horizontal position (e.g., catching the top of 'h' on the left but the middle of 'h' on the right).

* **Paper's Method (Baseline Detection):**
    The paper likely employs a **baseline detection** algorithm. This method identifies the imaginary line upon which the "body" of the letters sits (the baseline).
    * **Technical Detail:** It involves extracting the lower contour of the handwriting and fitting a line *only* to the local minima that correspond to the main body, explicitly identifying and ignoring "descenders" (tails of `g`, `y`, `p`, `q`).
    * **Why it works:** It aligns the core text regardless of how long the descenders are.

* **Your Implementation (`preprocess.py`):**
    Your `deslope_image` function uses **Global Linear Regression** on all ink pixels.
    * **Technical Detail:** `m, c = np.polyfit(x, y, 1)` calculates the line that minimizes the squared distance to *every* black pixel.
    * **The Flaw:** Descenders and ascenders are treated as equal to the body. A word like "jogging" (lots of tails) will statistically pull the regression line downwards, potentially causing the image to be rotated *incorrectly* to compensate for a slope that doesn't exist.

**Code Example: Robust Baseline Detection (Paper Style)**
```python
import numpy as np

def deslope_robust(img):
    """
    Fits a line ONLY to the bottom of the 'core' letters,
    ignoring descenders to find the true baseline.
    """
    h, w = img.shape
    # 1. Extract Lower Contour: Find lowest ink pixel in every column
    bottom_pixels = []
    for x in range(w):
        col = img[:, x]
        indices = np.where(col > 0)[0]
        if len(indices) > 0:
            bottom_pixels.append((x, np.max(indices))) # Max Y = Lowest pixel

    if not bottom_pixels: return img
    
    pts = np.array(bottom_pixels)
    X, Y = pts[:, 0], pts[:, 1]

    # 2. Iterative Regression with Outlier Rejection
    # We fit a line, then remove points significantly BELOW it (descenders)
    for _ in range(3):
        m, c = np.polyfit(X, Y, 1)
        predicted_y = m * X + c
        residuals = Y - predicted_y # Positive if point is below line
        
        # Keep points close to the line (the baseline core)
        # Discard points far below (descenders like 'g', 'y')
        valid_mask = residuals < 5.0 
        if np.sum(valid_mask) < 10: break
        X, Y = X[valid_mask], Y[valid_mask]

    angle = np.arctan(m) * (180 / np.pi)
    # ... apply rotation ...
    return angle
    
 ```
### Section 2: Decoding & Language Modeling

## 2. Decoding & Language Modeling

**The Challenge:** The HMM/ANN output is noisy. The model might predict "thue" instead of "the" because 'u' and 'e' look similar.

* **Paper's Method (Constrained Decoding):**
    The system relies on a **Language Model (LM)** or Lexicon.
    * **Technical Detail:** The Viterbi algorithm doesn't just transition between character states; it transitions between *word* states in a large Finite State Transducer (FST) or Trie.
    * **Mechanism:** If the ANN outputs high probability for 'z', but the current path forms "thz" (invalid word), the LM score ($P(word)$) drops to zero (or very low), forcing the Viterbi path to switch to the slightly less likely 'e' to form "the".

* **Your Implementation (`hmm.py`):**
    You use **Unconstrained Viterbi / Greedy Decoding**.
    * **Technical Detail:** Your `decode` function takes `argmax(log_probs)`. It selects the best character for *this specific frame* without caring about the previous or next character.
    * **The Flaw:** It has no concept of valid spelling. It produces phonetic gibberish because it cannot resolve ambiguity using context.

**Code Example: Dictionary-Constrained Decoding**
```python
def decode_with_dictionary(ann_output, vocabulary):
    """
    Simulated constrained decoding. Instead of argmax, we search
    for the valid word that maximizes the sequence probability.
    """
    # Simply put: We only allow transitions that exist in our dictionary Trie
    active_paths = {node: 0.0 for node in vocabulary.roots} 
    
    for t in range(len(ann_output)):
        next_paths = {}
        for node, current_score in active_paths.items():
            # Only consider characters that are valid next letters in the Trie
            for char, next_node in node.children.items():
                # Get probability of this char from ANN
                char_prob = ann_output[t, char_to_index[char]]
                
                new_score = current_score + char_prob
                if next_node not in next_paths or new_score > next_paths[next_node]:
                    next_paths[next_node] = new_score
        
        # Beam Search pruning (keep top 50 paths)
        active_paths = dict(sorted(next_paths.items(), key=lambda x: x[1])[-50:])
        
    # Return best path ending at a valid word
    return best_valid_word(active_paths)
```
### Section 3: Feature Extraction

## 3. Feature Extraction

**The Challenge:** Raw pixels are high-dimensional and noisy. We need a compact representation of the "ink shape."

* **Paper's Method (Geometric Features):**
    The paper cites using **9 geometric features** (Marti & Bunke, 2001) per window.
    * **Technical Detail:**
        1.  **0th Moment:** Fraction of black pixels (ink density).
        2.  **1st Moment:** Center of Gravity (vertical position of the ink).
        3.  **2nd Moment:** Inertia (how spread out the ink is).
        4.  **Contour Info:** Position of the upper-most and lower-most ink pixel.
        5.  **Gradients:** Changes in contour direction.

* **Your Implementation (`preprocess.py`):**
    You extract **3 derivative features** per cell.
    * **Technical Detail:** `val_n` (Gray Level), `val_h` (Horizontal Derivative), `val_v` (Vertical Derivative).
    * **Comparison:** Your features are good (derivatives capture edges well), but they lack the explicit *structural* information of moments. For example, "Center of Gravity" helps distinguishing 'e' (ink in middle) from 'l' (ink spreads up).

**Code Example: Geometric Moments**
```python
def get_moments(window):
    """
    Calculates 0th, 1st, and 2nd moments for a window column.
    """
    h = len(window)
    total_ink = np.sum(window)
    
    # 0th Moment: Weight / Density
    m0 = total_ink / h
    
    if total_ink == 0: return [m0, 0, 0]

    y_coords = np.arange(h)
    
    # 1st Moment: Center of Gravity (Mean Y)
    m1 = np.sum(y_coords * window) / total_ink
    
    # 2nd Moment: Variance (Spread)
    m2 = np.sum(((y_coords - m1) ** 2) * window) / total_ink
    
    return [m0, m1/h, m2/(h*h)] # Normalized
```
### Section 4: Neural Network Architecture

## 4. Neural Network Architecture

**The Challenge:** Map features to character probabilities.

* **Paper's Method (2011 Era MLP):**
    * **Layers:** Multilayer Perceptron.
    * **Activation:** **Sigmoid** or **Tanh**.
    * **Issue:** These activations suffer from the **Vanishing Gradient** problem. As the network gets deeper, gradients shrink to zero, making training slow or impossible for complex data.

* **Your Implementation (Modern MLP):**
    * **Layers:** `nn.Linear` with `nn.Dropout`.
    * **Activation:** **ReLU** (`nn.ReLU`).
    * **Normalization:** **Batch Normalization** (`nn.BatchNorm1d`).
    * **Advantage:** ReLU does not saturate (gradient is either 0 or 1), allowing faster and deeper learning. Batch Norm stabilizes the inputs to each layer. **This is one area where your implementation is technically superior to the original paper's.**

**Code Example: Architecture Comparison**
```python
# --- Paper Style (2011) ---
model = nn.Sequential(
    nn.Linear(540, 256),
    nn.Sigmoid(),        # Squashes values to [0,1], kills gradients
    nn.Linear(256, 128),
    nn.Sigmoid(),
    nn.Linear(128, num_classes)
)

# --- Your Implementation (Modern) ---
model = nn.Sequential(
    nn.Linear(540, 256),
    nn.BatchNorm1d(256), # Re-centers data to Mean=0, Std=1
    nn.ReLU(),           # Linear for positive values (No vanishing gradient)
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
)
```
### Section 5: HMM Topology (Space Modeling)

## 5. HMM Topology (Space Modeling)

**The Challenge:** "Space" is different from letters. Letters have a sequence (start->middle->end). Space is just "empty duration."

* **Paper's Method:**
    Usually employs a specific **"Silence" model**.
    * **Topology:** A single state with a high self-loop probability ($a_{ii}$) and transition probability to any character start state.
    * **Function:** It acts as the "glue" between words, absorbing the variable width of empty space between handwritten words.

* **Your Implementation (`dataset.py`):**
    You recently adopted a **Dynamic Topology**.
    * **Technical Detail:** `STATE_COUNTS = {' ': 1, 'm': 5, ...}`.
    * **Alignment:** By assigning `' '` (Space) exactly **1 state**, you effectively replicate the "Silence" model. It allows the HMM to stay in the Space state for 1 frame or 100 frames with equal ease, preventing the model from forcing "ghost characters" into empty spaces.

**Code Example: Topology Definition**
```python
# Defined in dataset.py
STATE_COUNTS = {
    ' ': 1,  # The "Silence" State (Loopable)
    'i': 2,  # Narrow letter (Short sequence)
    'm': 5,  # Wide letter (Long sequence)
}

# Used in hmm.py to build the Viterbi path
def char_to_state_seq(char):
    count = STATE_COUNTS.get(char, 3)
    # Returns [10] for Space, but [20, 21, 22, 23, 24] for 'm'
    return [start + i for i in range(count)]