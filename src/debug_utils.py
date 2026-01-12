import torch
import numpy as np
from dataset import CHAR_TO_STATE_RANGES


def raw_dump(output_tensor):
    """
    Takes the output (Batch, Time, Classes) and prints the top prediction
    for each time step.
    Collapses by CHARACTER, not by state ID.
    """
    probs = output_tensor.squeeze(0).cpu().detach().numpy()
    preds = np.argmax(probs, axis=1)

    state_to_char = {}
    for char, (start, count) in CHAR_TO_STATE_RANGES.items():
        for i in range(count):
            state_to_char[start + i] = char

    # Collapse by CHARACTER (not state)
    collapsed = []
    prev_char = None
    for idx in preds:
        char = state_to_char.get(idx, '~')
        if char != prev_char:
            collapsed.append(char)
            prev_char = char

    return "".join(collapsed)


def raw_dump_with_states(output_tensor):
    """
    Debug version that shows both states and characters.
    Returns (collapsed_text, raw_state_sequence, state_counts)
    """
    probs = output_tensor.squeeze(0).cpu().detach().numpy()
    preds = np.argmax(probs, axis=1)

    state_to_char = {}
    for char, (start, count) in CHAR_TO_STATE_RANGES.items():
        for i in range(count):
            state_to_char[start + i] = char

    # Count consecutive states
    state_runs = []
    if len(preds) > 0:
        current_state = preds[0]
        count = 1
        for i in range(1, len(preds)):
            if preds[i] == current_state:
                count += 1
            else:
                state_runs.append((current_state, count))
                current_state = preds[i]
                count = 1
        state_runs.append((current_state, count))

    # Collapse by character
    collapsed = []
    prev_char = None
    for idx in preds:
        char = state_to_char.get(idx, '~')
        if char != prev_char:
            collapsed.append(char)
            prev_char = char

    return "".join(collapsed), state_runs
