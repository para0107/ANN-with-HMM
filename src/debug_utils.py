import torch
import numpy as np
from dataset import CHAR_TO_STATE_RANGES

def raw_dump(output_tensor):
    """
    Takes the output (Batch, Time, Classes) and prints the top prediction
    for each time step.
    Updated for Dynamic Topology: Uses state ranges to map IDs back to Chars.
    """
    probs = output_tensor.squeeze(0).cpu().detach().numpy()
    preds = np.argmax(probs, axis=1)

    state_to_char = {}
    for char, (start, count) in CHAR_TO_STATE_RANGES.items():
        for i in range(count):
            state_to_char[start + i] = char

    res = []
    for idx in preds:
        char = state_to_char.get(idx, '~')
        res.append(char)

    collapsed = []
    prev = None
    for c in res:
        if c != prev:
            collapsed.append(c)
            prev = c

    return "".join(collapsed)