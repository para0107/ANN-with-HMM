import torch
import numpy as np
from dataset import CHARS, STATES_PER_CHAR


def raw_dump(output_tensor):
    """
    Takes the output (Batch, Time, Classes) and prints the top prediction
    for each time step.
    """
    # Output shape: (1, Time, Classes)
    probs = output_tensor.squeeze(0).cpu().detach().numpy()  # (Time, Classes)
    preds = np.argmax(probs, axis=1)

    # Convert indices to characters
    res = []
    for idx in preds:
        char_idx = idx // STATES_PER_CHAR
        if char_idx < len(CHARS):
            res.append(CHARS[char_idx])
        else:
            res.append('~')  # Unknown/Garbage

    # Collapse repeats for readability
    collapsed = []
    prev = None
    for c in res:
        if c != prev:
            collapsed.append(c)
            prev = c

    return "".join(collapsed)