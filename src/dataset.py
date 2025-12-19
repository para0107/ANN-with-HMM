import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os
#define the characters and how they map to states
#7 states per character(recommended in the paper)
#load feature vectors and parse the XML labels

# --- Configuration ---
CHARS = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
STATES_PER_CHAR = 3
TOTAL_STATES = len(CHARS) * STATES_PER_CHAR


def char_to_state_id(char):
    if char not in CHARS:
        char = ' '  # Map unknown chars to space
    return CHARS.index(char) * STATES_PER_CHAR


def text_to_flat_start_path(text, total_frames):
    """Evenly distributes frames among characters (Flat Start)."""
    sequence_states = []
    for char in text:
        start = char_to_state_id(char)
        for i in range(STATES_PER_CHAR):
            sequence_states.append(start + i)

    num_states = len(sequence_states)
    if num_states == 0: return np.zeros(total_frames, dtype=int)

    # Safety: if image is shorter than minimum states
    if total_frames < num_states:
        # Just repeat the first state to fill (bad, but prevents crash)
        return np.zeros(total_frames, dtype=int)

    # Linear interpolation
    indices = np.linspace(0, num_states, total_frames, endpoint=False).astype(int)
    return np.array([sequence_states[i] for i in indices])


def get_transcription(xml_dir, line_id):
    """
    Parses IAM XML to find text for a line.
    ID Format: 'a01-007-00' -> file 'a01-007.xml'
    """
    parts = line_id.split('-')
    if len(parts) < 2: return ""

    form_id = f"{parts[0]}-{parts[1]}"  # e.g. a01-007
    xml_path = os.path.join(xml_dir, form_id + ".xml")

    if not os.path.exists(xml_path):
        return ""

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
        """
        Scans feature_dir for .npy files and matches them with XML labels.
        """
        self.feature_dir = feature_dir
        self.xml_dir = xml_dir
        self.window_width = window_width
        self.half_window = window_width // 2

        self.data_entries = []  # List of line_ids
        self.target_cache = {}  # Stores Viterbi alignments

        # 1. Discover files
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

        print(f"Scanning {feature_dir}...")
        files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]

        # 2. Filter valid lines (those that have text)
        valid_count = 0
        for f in files:
            line_id = f.replace('.npy', '')
            text = get_transcription(xml_dir, line_id)
            if text:
                self.data_entries.append(line_id)
                valid_count += 1

        print(f"Found {valid_count} valid lines.")

    def __len__(self):
        return len(self.data_entries)

    def update_target_at_index(self, idx, new_target):
        """Cache the new Viterbi alignment target."""
        self.target_cache[idx] = new_target

    def get_item_with_text(self, idx):
        """Helper to get raw data + text for alignment."""
        line_id = self.data_entries[idx]
        text = get_transcription(self.xml_dir, line_id)

        # Load Features
        feat_path = os.path.join(self.feature_dir, line_id + ".npy")
        features = np.load(feat_path).astype(np.float32)

        # Pad and Window
        features_padded = np.pad(features, ((self.half_window, self.half_window), (0, 0)), mode='edge')
        num_frames = features.shape[0]
        feat_dim = features.shape[1]

        # Efficient sliding window
        # Shape: (T, Window*Dim) -> (T, 540)
        windows = np.zeros((num_frames, self.window_width * feat_dim), dtype=np.float32)
        for t in range(num_frames):
            win = features_padded[t: t + self.window_width]
            windows[t] = win.flatten()

        return torch.from_numpy(windows), None, text

    def __getitem__(self, idx):
        windows, _, text = self.get_item_with_text(idx)

        # Determine Target: Use Cache (Viterbi) if available, else Flat Start
        if idx in self.target_cache:
            targets = self.target_cache[idx]
        else:
            targets = torch.from_numpy(text_to_flat_start_path(text, windows.shape[0])).long()

        return windows, targets, text