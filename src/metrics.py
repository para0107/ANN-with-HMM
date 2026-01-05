import numpy as np

def levenshtein(a, b):
    """Calculates the Levenshtein distance between two strings."""
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]

def calculate_cer(predicted_str, target_str):
    """Returns the Character Error Rate (0.0 to 1.0+)."""
    if len(target_str) == 0:
        return 1.0 if len(predicted_str) > 0 else 0.0
    dist = levenshtein(predicted_str, target_str)
    return dist / len(target_str)