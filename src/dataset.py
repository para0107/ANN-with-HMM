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
    # Padding with spaces to handle margins
    padded_text = "   " + text + "   "

    full_state_sequence = []
    for char in padded_text:
        full_state_sequence.extend(char_to_state_seq(char))

    num_states = len(full_state_sequence)
    if num_states == 0: return np.zeros(total_frames, dtype=int)

    if total_frames < num_states:
        return np.array(full_state_sequence[:total_frames])

    indices = np.linspace(0, num_states, total_frames, endpoint=False).astype(int)
    return np.array([full_state_sequence[i] for i in indices])


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


# --- NEW: Custom Collate Function ---
def iam_collate_fn(batch):
    """
    Pads features and targets to the max length in the batch.
    """
    features, targets, texts = zip(*batch)

    # 1. Pad Features (Batch, Max_Time, Dim)
    # pad_sequence expects list of (Time, Dim)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    # 2. Pad Targets (Batch, Max_Time)
    # We use -1 as the ignore index (loss function must know this)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-1)

    return padded_features, padded_targets, texts