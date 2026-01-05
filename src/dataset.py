import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os

# --- Configuration ---
CHARS = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
STATES_PER_CHAR = 3
TOTAL_STATES = len(CHARS) * STATES_PER_CHAR


def char_to_state_id(char):
    if char not in CHARS:
        char = ' '
    return CHARS.index(char) * STATES_PER_CHAR


def state_id_to_char(state_id):
    if state_id < 0 or state_id >= TOTAL_STATES:
        return ''
    char_idx = state_id // STATES_PER_CHAR
    return CHARS[char_idx]


def text_to_flat_start_path(text, total_frames):
    """
    Standard Flat Start: Distributes text evenly across the image.
    (Padding removed to prevent 'Space Collapse')
    """
    sequence_states = []
    for char in text:
        start = char_to_state_id(char)
        for i in range(STATES_PER_CHAR):
            sequence_states.append(start + i)

    num_states = len(sequence_states)
    if num_states == 0: return np.zeros(total_frames, dtype=int)

    # Safety: if image is shorter than minimum states required
    if total_frames < num_states:
        return np.array([sequence_states[0]] * total_frames)

    # Linear interpolation
    indices = np.linspace(0, num_states, total_frames, endpoint=False).astype(int)
    return np.array([sequence_states[i] for i in indices])


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
            if line.get('id') == line_id:
                return line.get('text')
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

        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Feature dir not found: {feature_dir}")

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