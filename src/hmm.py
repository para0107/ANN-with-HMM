import numpy as np
from dataset import CHARS, STATES_PER_CHAR


class HybridHMM:
    def __init__(self, num_chars=len(CHARS), states_per_char=STATES_PER_CHAR, num_classes=None):
        if num_classes is not None:
            self.total_states = num_classes
        else:
            self.total_states = num_chars * states_per_char

        # --- 1. PRIORS ---
        self.priors = np.full(self.total_states, 1.0 / self.total_states)
        self.prior_counts = np.zeros(self.total_states)

        # --- 2. TRANSITIONS ---
        self.transitions = np.full((self.total_states, 2), 0.5)
        self.trans_counts = np.zeros((self.total_states, 2))

    def reset_accumulators(self):
        self.prior_counts.fill(0)
        self.trans_counts.fill(0)

    def update_parameters(self):
        total_frames = np.sum(self.prior_counts)
        if total_frames > 0:
            self.priors = self.prior_counts / total_frames

        row_sums = np.sum(self.trans_counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transitions = self.trans_counts / row_sums

    def get_scaled_emissions(self, ann_log_output):
        log_priors = np.log(self.priors + 1e-10)
        return ann_log_output - log_priors

    def decode(self, log_probs):
        """
        Greedy Decoding: Best state at each time step -> Character.
        Used for inference/prediction.
        """
        # 1. Get best state index for each time step
        # Shape: (Time, States) -> (Time,)
        best_states = np.argmax(log_probs, axis=1)

        decoded_text = []
        prev_char_idx = -1

        # 2. Collapse states into characters
        for state in best_states:
            char_idx = state // STATES_PER_CHAR

            # Simple heuristic: Only append if it's a new character
            # or the state indicates the start of a new character sequence.
            # (Here we use a simple collapse: repeated char indices are ignored)
            if char_idx != prev_char_idx:
                if char_idx < len(CHARS):
                    decoded_text.append(CHARS[char_idx])
                prev_char_idx = char_idx

        return "".join(decoded_text)

    def forced_alignment(self, scaled_emissions, text_state_indices):
        """Viterbi Forced Alignment (Same as before)"""
        T = scaled_emissions.shape[0]
        S = len(text_state_indices)
        scores = np.full((T, S), -np.inf)
        backpointers = np.zeros((T, S), dtype=int)
        log_trans = np.log(self.transitions + 1e-10)

        if S > 0:
            scores[0, 0] = scaled_emissions[0, text_state_indices[0]]

        for t in range(1, T):
            for s in range(S):
                curr = text_state_indices[s]
                emission = scaled_emissions[t, curr]

                # Stay
                score_stay = scores[t - 1, s] + log_trans[curr, 0]

                # Move
                score_move = -np.inf
                if s > 0:
                    prev = text_state_indices[s - 1]
                    score_move = scores[t - 1, s - 1] + log_trans[prev, 1]

                if score_stay > score_move:
                    scores[t, s] = score_stay + emission
                    backpointers[t, s] = 0
                else:
                    scores[t, s] = score_move + emission
                    backpointers[t, s] = 1

        path = np.zeros(T, dtype=int)
        if S > 0:
            curr_s = S - 1
            if scores[T - 1, curr_s] == -np.inf: return None

            for t in range(T - 1, -1, -1):
                path[t] = text_state_indices[curr_s]
                self.prior_counts[path[t]] += 1
                if t > 0:
                    if backpointers[t, curr_s] == 0:
                        self.trans_counts[path[t], 0] += 1
                    else:
                        prev_global = text_state_indices[curr_s - 1]
                        self.trans_counts[prev_global, 1] += 1
                        curr_s -= 1
        return path