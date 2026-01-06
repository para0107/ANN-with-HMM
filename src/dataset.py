import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import xml.etree.ElementTree as ET
import os

# --- Configuration ---
CHARS = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# --- DYNAMIC STATE MAPPING ---
STATE_COUNTS = {
    ' ': 1,  # Space
    '.': 2, ',': 2, "'": 2, '-': 2, '!': 2, 'i': 2, 'I': 2, 'l': 2, 'j': 2, '1': 2,
    'm': 5, 'w': 5, 'M': 5, 'W': 5,
    'f': 4, 't': 3, 'r': 3,
}
DEFAULT_STATES = 3
FRAMES_PER_STATE = 4  # Heuristic: How many visual frames typically make up one state?

CHAR_TO_STATE_RANGES = {}
TOTAL_STATES = 0

for char in CHARS:
    count = STATE_COUNTS.get(char, DEFAULT_STATES)
    start = TOTAL_STATES
    CHAR_TO_STATE_RANGES[char] = (start, count)
    TOTAL_STATES += count


def char_to_state_seq(char):
    if char not in CHARS: char = ' '
    start, count = CHAR_TO_STATE_RANGES[char]
    return [start + i for i in range(count)]


def text_to_flat_start_path(text, total_frames):
    """
    Centered Flat Start:
    Estimates the length of the text and centers it in the image,
    filling the surrounding area with Space states.
    Prevents 'Stretching' which poisons the model.
    """
    # 1. Generate the sequence of states for the text
    text_state_sequence = []
    for char in text:
        text_state_sequence.extend(char_to_state_seq(char))

    num_text_states = len(text_state_sequence)
    if num_text_states == 0: return np.zeros(total_frames, dtype=int)

    # 2. Estimate required frames (Heuristic)
    # We repeat each state N times to give it duration
    estimated_len = num_text_states * FRAMES_PER_STATE

    # 3. Create the full path array initialized to Space (ID for ' ' state)
    # Find ID for space
    space_start, _ = CHAR_TO_STATE_RANGES[' ']
    path = np.full(total_frames, space_start, dtype=int)

    # 4. Handle Lengths
    if estimated_len >= total_frames:
        # Text is too long (or image too short), squeeze it to fit
        indices = np.linspace(0, num_text_states, total_frames, endpoint=False).astype(int)
        path = np.array([text_state_sequence[i] for i in indices])
    else:
        # Text fits inside. Center it.
        start_idx = (total_frames - estimated_len) // 2

        # Expand the text states
        expanded_seq = []
        for s in text_state_sequence:
            expanded_seq.extend([s] * FRAMES_PER_STATE)

        # Copy into the middle of the path
        path[start_idx: start_idx + len(expanded_seq)] = expanded_seq

    return path


def get_transcription(xml_dir, line_id):
    parts = line_id.split('-')
    if len(parts) < 2: return ""
    form_id = f"{parts[0]}-{parts[1]}"
    xml_path = os.path.join(xml_dir, form_id + ".xml")
    if not os.path.exists(xml_path): return ""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for line in root.findall(".//line"):
            if line.get('id') == line_id: return line.get('text')
    except:
        pass
    return ""


class IAMDataset(Dataset):
    def __init__(self, feature_dir, xml_dir, window_width=9):
        self.feature_dir = feature_dir
        self.xml_dir = xml_dir
        self.window_width = window_width
        self.half_window = window_width // 2
        self.data_entries = []
        self.target_cache = {}

        if not os.path.exists(feature_dir): return
        files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        for f in files:
            line_id = f.replace('.npy', '')
            text = get_transcription(xml_dir, line_id)
            if text: self.data_entries.append(line_id)

    def __len__(self):
        return len(self.data_entries)

    def update_target_at_index(self, idx, new_target):
        self.target_cache[idx] = new_target

    def get_item_with_text(self, idx):
        line_id = self.data_entries[idx]
        text = get_transcription(self.xml_dir, line_id)
        feat_path = os.path.join(self.feature_dir, line_id + ".npy")
        try:
            features = np.load(feat_path).astype(np.float32)
        except:
            return torch.zeros((10, 540)), None, ""

        features_padded = np.pad(features, ((self.half_window, self.half_window), (0, 0)), mode='edge')
        num_frames = features.shape[0]
        feat_dim = features.shape[1]
        windows = np.zeros((num_frames, self.window_width * feat_dim), dtype=np.float32)
        for t in range(num_frames):
            win = features_padded[t: t + self.window_width]
            windows[t] = win.flatten()
        return torch.from_numpy(windows), None, text

    def __getitem__(self, idx):
        windows, _, text = self.get_item_with_text(idx)
        if idx in self.target_cache:
            targets = self.target_cache[idx]
        else:
            targets = torch.from_numpy(text_to_flat_start_path(text, windows.shape[0])).long()
        return windows, targets, text


def iam_collate_fn(batch):
    features, targets, texts = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)
    return features_padded, targets_padded, texts