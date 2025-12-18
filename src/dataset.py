#define the characters and how they map to states
#7 states per character(recommended in the paper)
#load feature vectors and parse the XML labels
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os

# 78 characters from the IAM
characters = '!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
states_per_character = 7

# number of classes to clasify an character(total states in HMM)
states = len(characters) * states_per_character
print(f"Correct number of states is 546. We have: {states}")

def char_to_state_id(char):
    """Maps a character to its state index"""
    if char not in characters:
        #handle unknown characters
        char = ' '
    char_index = characters.index(char)
    start_state = char_index*states_per_character
    return start_state

def text_to_flat_start_path(text, total_frames):
    """
        Generates a 'Flat Start' alignment.
        Evenly distributes the image frames among the characters in the text.

        Args:
            text (str): The label (e.g., "The")
            total_frames (int): Number of feature vectors in the image.

        Returns:
            np.array: Array of size (total_frames,) containing state IDs.
        """
    # 1. Convert text to a list of ALL states required(eg. "the"-> states for t, h and e
    sequence_states= []
    for char in text:
        start_state = char_to_state_id(char)
        for i in range(states_per_character):
            sequence_states.append(start_state+i)#Append all 7 states of a character
    num_states_in_sequence = len(sequence_states)
    if total_frames < num_states_in_sequence: # image too short for the text
        print(f"Image too short ({total_frames}) for text '{text}' ({num_states_in_sequence} states).")
        return np.zeros(total_frames, dtype=int)

    path = np.zeros(total_frames, dtype=int)

    #find transition indices
    indices = np.linspace(0, num_states_in_sequence, total_frames, endpoint=False).astype(int)

    for t in range(total_frames):
        state_index = indices[t]
        path[t] = sequence_states[state_index]
    return path


def parse_iam_xml_line(xml_path, line_id):
    """
    Parses the XML file to find the transcription for a specific line_id.
    Ex: line_id = "a01-007u-00"
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Iterate over all handwritten lines
        for line in root.findall(".//line"):
            if line.get('id') == line_id:
                # Reconstruct text from words to preserve spacing if needed,
                # or usually just take the 'text' attribute directly if available.
                # IAM XML structure usually has 'text' attribute on the line tag
                return line.get('text')

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None
    return None


class IAMDataset(Dataset):
    def __init__(self, root_dir, feature_dir, split_file="lines.txt", window_width=9):
        """
        Args:
            root_dir (str): Path to data folder.
            feature_dir (str): Path where .npy files are stored.
            split_file (str): Path to the train/test split file.
            window_width (int): Total context window size (default 9 from paper).
        """
        self.feature_dir = feature_dir
        self.window_width = window_width
        self.half_window = window_width // 2

        # Load list of file IDs from the split file (e.g., train.txt)
        # Assumes lines are formatted like: "a01-007u-00 OK ..."
        self.data_entries = []

        with open(os.path.join(root_dir, split_file), 'r') as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) > 8:
                    line_id = parts[0]  # e.g., "a01-007u-00"

                    # Need to find the original text.
                    # Option A: Parse XML every time (Slow)
                    # Option B: Parse once and store in a list (Better)
                    # For this snippet, we assume we pass the text or parse it here.
                    # Let's assume you have a helper to get text from ID:
                    # text = get_text_for_id(line_id)

                    # Store tuple: (line_id, text_label)
                    # NOTE: You need to implement get_text_from_xml logic here
                    self.data_entries.append(line_id)

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        line_id = self.data_entries[idx]

        # 1. Load Features
        feat_path = os.path.join(self.feature_dir, line_id + ".npy")
        # Shape: (Time, 60)
        features = np.load(feat_path)

        # 2. Apply Sliding Window
        # We need to pad the features so we can center the window on every frame
        # Pad width: (4, 4) for dim 0, (0,0) for dim 1
        features_padded = np.pad(features, ((self.half_window, self.half_window), (0, 0)), mode='edge')

        # Create sliding windows efficiently
        # Final Shape: (Time, 540)
        num_frames = features.shape[0]
        feature_dim = features.shape[1]

        windows = np.zeros((num_frames, self.window_width * feature_dim), dtype=np.float32)

        for t in range(num_frames):
            # Extract window of 9 frames
            win = features_padded[t: t + self.window_width]  # Shape (9, 60)
            windows[t] = win.flatten()  # Shape (540,)

        # 3. Load Text and Create Target (Flat Start)
        # Note: In real training, you'd look up the text.
        text = "placeholder"

        targets = text_to_flat_start_path(text, num_frames)

        return torch.from_numpy(windows), torch.from_numpy(targets)

    def get_transcription(xml_dir, line_id):
        """
        Finds the text label for a given line ID using the IAM XML structure.

        Args:
            xml_dir (str): Path to the folder containing xml files.
            line_id (str): The ID of the line, e.g., "a01-007-00".

        Returns:
            str: The transcription text (e.g., "Since 1958...")
        """
        # 1. Deduce the XML filename from the line_id
        # Format: a01-007-00 -> form ID is a01-007
        parts = line_id.split('-')
        form_id = f"{parts[0]}-{parts[1]}"  # "a01-007"
        xml_path = os.path.join(xml_dir, form_id + ".xml")

        if not os.path.exists(xml_path):
            print(f"Warning: XML file not found for {line_id}")
            return None

        try:
            # 2. Parse the XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 3. Find the specific line tag
            # We look for <line id="a01-007-00" ... >
            for line in root.findall(".//line"):
                if line.get('id') == line_id:
                    # The text is in the 'text' attribute
                    return line.get('text')

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None

        return None




