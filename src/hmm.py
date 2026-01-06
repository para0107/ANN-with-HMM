import numpy as np
from dataset import CHAR_TO_STATE_RANGES, TOTAL_STATES, CHARS


class HybridHMM:
    def __init__(self, num_classes=TOTAL_STATES):
        self.total_states = num_classes
        self.priors = np.full(self.total_states, 1.0 / self.total_states)
        self.prior_counts = np.zeros(self.total_states)
        self.transitions = np.zeros((self.total_states, 2))
        self.transitions[:, 0] = 0.9  # 90% probability to stay
        self.transitions[:, 1] = 0.1  # 10% probability to move

        self.trans_counts = np.zeros((self.total_states, 2))

    def reset_accumulators(self):
        self.prior_counts.fill(0)
        self.trans_counts.fill(0)

    def update_parameters(self):
        total_frames = np.sum(self.prior_counts)
        if total_frames > 0: self.priors = self.prior_counts / total_frames
        row_sums = np.sum(self.trans_counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transitions = self.trans_counts / row_sums

    def get_scaled_emissions(self, ann_log_output):
        # ann_log_output shape: (Time, States)

        # 1. Standard Bayes Rule
        log_priors = np.log(self.priors + 1e-10)
        scaled = ann_log_output - log_priors

        # 2. HEURISTIC: Penalize the 'Space' state slightly to force characters
        # Assuming State 0 is space (check CHAR_TO_STATE_RANGES to be sure)
        space_idx = 0
        # Add a penalty (negative value) to space log-prob
        # Or simply boost all others
        scaled[:, space_idx] -= 1.0

        return scaled

    def decode(self, log_probs):
        """Greedy Decode handling variable states."""
        best_states = np.argmax(log_probs, axis=1)
        decoded_text = []
        prev_char = None

        # Build Reverse Map: State ID -> Char
        state_to_char = {}
        for char, (start, count) in CHAR_TO_STATE_RANGES.items():
            for i in range(count):
                state_to_char[start + i] = char

        for state in best_states:
            if state in state_to_char:
                char = state_to_char[state]
                if char != prev_char:
                    decoded_text.append(char)
                    prev_char = char
        return "".join(decoded_text)

    def forced_alignment(self, scaled_emissions, text):
        """Viterbi for Dynamic States."""
        T = scaled_emissions.shape[0]

        # Build sequence of states allowed for this specific text string
        allowable_states = []
        for char in text:
            if char in CHAR_TO_STATE_RANGES:
                start, count = CHAR_TO_STATE_RANGES[char]
                for i in range(count): allowable_states.append(start + i)

        S = len(allowable_states)
        if S == 0: return None

        scores = np.full((T, S), -np.inf)
        backpointers = np.zeros((T, S), dtype=int)
        log_trans = np.log(self.transitions + 1e-10)

        # Init
        start_global = allowable_states[0]
        scores[0, 0] = scaled_emissions[0, start_global]

        for t in range(1, T):
            for s in range(S):
                curr_global = allowable_states[s]
                emission = scaled_emissions[t, curr_global]

                # Stay
                score_stay = scores[t - 1, s] + log_trans[curr_global, 0]

                # Move
                score_move = -np.inf
                if s > 0:
                    prev_global = allowable_states[s - 1]
                    score_move = scores[t - 1, s - 1] + log_trans[prev_global, 1]

                if score_stay > score_move:
                    scores[t, s] = score_stay + emission
                    backpointers[t, s] = 0
                else:
                    scores[t, s] = score_move + emission
                    backpointers[t, s] = 1

        path = np.zeros(T, dtype=int)
        curr_s = S - 1
        if scores[T - 1, curr_s] == -np.inf: return None

        for t in range(T - 1, -1, -1):
            global_id = allowable_states[curr_s]
            path[t] = global_id
            self.prior_counts[global_id] += 1
            if t > 0:
                if backpointers[t, curr_s] == 0:
                    self.trans_counts[global_id, 0] += 1
                else:
                    prev_global_s = allowable_states[curr_s - 1]
                    self.trans_counts[prev_global_s, 1] += 1
                    curr_s -= 1
        return path