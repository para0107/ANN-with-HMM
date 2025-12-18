import numpy as np
import matplotlib.pyplot as plt


def levenshtein(seq1, seq2):
    """
    Calculates the Edit Distance between two sequences (strings or lists).
    Used for both CER (chars) and WER (words).
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x): matrix[x, 0] = x
    for y in range(size_y): matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1  # Insertion
                )
    return matrix[size_x - 1, size_y - 1]


def calculate_error_rates(predictions, ground_truths):
    """
    Args:
        predictions: List of predicted strings (e.g. ["the cat", "hello"])
        ground_truths: List of reference strings.
    Returns:
        cer: Character Error Rate (0.0 to 1.0+)
        wer: Word Error Rate (0.0 to 1.0+)
    """
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0

    for pred, ref in zip(predictions, ground_truths):
        # CER Calculation
        total_char_dist += levenshtein(pred, ref)
        total_chars += len(ref)

        # WER Calculation (split by space)
        pred_words = pred.split()
        ref_words = ref.split()
        total_word_dist += levenshtein(pred_words, ref_words)
        total_words += len(ref_words)

    cer = total_char_dist / total_chars if total_chars > 0 else 0
    wer = total_word_dist / total_words if total_words > 0 else 0

    return cer, wer


from dataset import characters, states_per_character


def greedy_decode(ann_output_log_probs):
    """
    Takes ANN output (Time, 546) and converts to string.
    Rule:
    1. Take max probability state at each frame.
    2. Collapse repeats (State 0, State 0 -> State 0).
    3. Map State ID -> Character.
    """
    # 1. Argmax: Get best state for every frame
    best_states = np.argmax(ann_output_log_probs, axis=1)

    # 2. Collapse repeats & Map to chars
    decoded_str = []
    last_state = -1

    for state in best_states:
        if state != last_state:
            # We only record the character when we enter a NEW state
            # (Simple heuristic; real HMM decoding is better)
            char_idx = state // states_per_character
            if char_idx < len(characters):
                decoded_str.append(characters[char_idx])
            last_state = state

    return "".join(decoded_str)


def plot_training_history(history, save_path="training_plot.png"):
    """
    Plots Loss vs Epochs on the left axis, and WER/CER on the right axis.

    Args:
        history: Dict containing lists: 'loss', 'val_loss', 'cer', 'wer'
    """
    epochs = range(len(history['loss']))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Axis: LOSS
    color = 'tab:red'
    ax1.set_xlabel('Epochs (EM Cycles)')
    ax1.set_ylabel('NLL Loss', color=color)
    ax1.plot(epochs, history['loss'], color=color, linestyle='-', label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], color=color, linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Right Axis: ERROR RATES
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Error Rate (%)', color=color)

    if 'cer' in history:
        ax2.plot(epochs, [x * 100 for x in history['cer']], color='tab:blue', label='CER')
    if 'wer' in history:
        ax2.plot(epochs, [x * 100 for x in history['wer']], color='tab:green', label='WER')

    ax2.tick_params(axis='y', labelcolor=color)

    # Title & Layout
    plt.title('Hybrid HMM/ANN Training Progress')
    fig.tight_layout()

    # Combined Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center')

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")