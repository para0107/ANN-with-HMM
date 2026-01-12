import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import xml.etree.ElementTree as ET
import os

CHARS = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

STATE_COUNTS = {
    ' ': 1,
    '.': 2, ',': 2, "'": 2, '-': 2, '!': 2, 'i': 2, 'I': 2, 'l': 2, 'j': 2, '1': 2,
    'm': 5, 'w': 5, 'M': 5, 'W': 5,
    'f': 4, 't': 3, 'r': 3,
}
DEFAULT_STATES = 3
FRAMES_PER_STATE = 2

CHAR_WEIGHTS = {
    'm': 1.5, 'w': 1.5, 'M': 1.5, 'W': 1.5,
    'i': 0.6, 'l': 0.6, 'I': 0.6, '1': 0.6, 'j': 0.6,
    '.': 0.4, ',': 0.4, "'": 0.4,
    ' ': 0.3,
}

CHAR_TO_STATE_RANGES = {}
TOTAL_STATES = 0

for char in CHARS:
    count = STATE_COUNTS.get(char, DEFAULT_STATES)
    start = TOTAL_STATES
    CHAR_TO_STATE_RANGES[char] = (start, count)
    TOTAL_STATES += count


def char_to_state_seq(char):
    if char not in CHARS:
        char = ' '
    start, count = CHAR_TO_STATE_RANGES[char]
    return [start + i for i in range(count)]


def text_to_flat_start_path(text, total_frames):
    """Improved flat-start with proportional allocation based on character width."""
    if not text:
        space_start, _ = CHAR_TO_STATE_RANGES[' ']
        return np.full(total_frames, space_start, dtype=int)

    states_with_weights = []
    for char in text:
        if char not in CHAR_TO_STATE_RANGES:
            char = ' '
        start, count = CHAR_TO_STATE_RANGES[char]
        weight = CHAR_WEIGHTS.get(char, 1.0)
        for i in range(count):
            states_with_weights.append((start + i, weight / count))

    if not states_with_weights:
        space_start, _ = CHAR_TO_STATE_RANGES[' ']
        return np.full(total_frames, space_start, dtype=int)

    total_weight = sum(w for _, w in states_with_weights)

    path = []
    accumulated_frames = 0.0

    for state, weight in states_with_weights:
        num_frames = (weight / total_weight) * total_frames
        accumulated_frames += num_frames
        target_frames = int(round(accumulated_frames)) - len(path)
        target_frames = max(1, target_frames)
        path.extend([state] * target_frames)

    while len(path) < total_frames:
        path.append(path[-1] if path else 0)
    path = path[:total_frames]

    return np.array(path, dtype=int)


def get_transcription(xml_dir, line_id):
    parts = line_id.split('-')
    if len(parts) < 2:
        return ""
    form_id = f"{parts[0]}-{parts[1]}"
    xml_path = os.path.join(xml_dir, form_id + ".xml")
    if not os.path.exists(xml_path):
        return ""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for line in root.findall(".//line"):
            if line.get('id') == line_id:
                text = line.get('text', '')
                return text.replace('&quot;', '"').replace('&apos;', "'").replace('&amp;', '&')
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
        self.feature_stats = None

        if not os.path.exists(feature_dir):
            return
        files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        for f in files:
            line_id = f.replace('.npy', '')
            text = get_transcription(xml_dir, line_id)
            if text:
                self.data_entries.append(line_id)

    def __len__(self):
        return len(self.data_entries)

    def update_target_at_index(self, idx, new_target):
        self.target_cache[idx] = new_target

    def clear_target_cache(self):
        self.target_cache.clear()

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

        mean = windows.mean()
        std = windows.std() + 1e-6
        windows = (windows - mean) / std

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
