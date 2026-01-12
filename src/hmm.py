import numpy as np
from dataset import CHAR_TO_STATE_RANGES, TOTAL_STATES, CHARS


class HybridHMM:
    def __init__(self, num_classes=TOTAL_STATES, min_state_duration=1):
        self.total_states = num_classes
        self.min_state_duration = min_state_duration
        self.priors = np.full(self.total_states, 1.0 / self.total_states)
        self.prior_counts = np.zeros(self.total_states)

        self.transitions = np.zeros((self.total_states, 2))
        self.transitions[:, 0] = 0.7
        self.transitions[:, 1] = 0.3
        self.trans_counts = np.zeros((self.total_states, 2))

        self.state_to_char = {}
        for char, (start, count) in CHAR_TO_STATE_RANGES.items():
            for i in range(count):
                self.state_to_char[start + i] = char

    def reset_accumulators(self):
        self.prior_counts.fill(0)
        self.trans_counts.fill(0)

    def update_parameters(self, smoothing=1.0):
        total_frames = np.sum(self.prior_counts) + smoothing * self.total_states
        self.priors = (self.prior_counts + smoothing) / total_frames

        self.priors = np.clip(self.priors, 1e-6, 1.0)
        self.priors /= self.priors.sum()

        row_sums = np.sum(self.trans_counts, axis=1, keepdims=True) + 2 * smoothing
        self.transitions = (self.trans_counts + smoothing) / row_sums

        self.transitions = np.clip(self.transitions, 0.05, 0.95)

    def get_scaled_emissions(self, ann_log_output, penalize_space=False):
        log_priors = np.log(self.priors + 1e-10)
        scaled = ann_log_output - log_priors

        if penalize_space:
            space_idx = 0
            scaled[:, space_idx] -= 0.5

        return scaled

    def decode(self, log_probs, penalize_space=True):
        if penalize_space:
            log_probs = log_probs.copy()
            log_probs[:, 0] -= 0.5

        best_states = np.argmax(log_probs, axis=1)
        decoded_text = []
        prev_char = None

        for state in best_states:
            if state in self.state_to_char:
                char = self.state_to_char[state]
                if char != prev_char:
                    decoded_text.append(char)
                    prev_char = char
        return "".join(decoded_text)

    def forced_alignment(self, scaled_emissions, text):
        T = scaled_emissions.shape[0]

        allowable_states = []
        for char in text:
            if char in CHAR_TO_STATE_RANGES:
                start, count = CHAR_TO_STATE_RANGES[char]
                for i in range(count):
                    allowable_states.append(start + i)

        S = len(allowable_states)
        if S == 0:
            return None

        min_frames_needed = S * self.min_state_duration
        if T < min_frames_needed:
            return self._proportional_alignment(text, T)

        scores = np.full((T, S), -np.inf)
        backpointers = np.zeros((T, S), dtype=int)
        duration = np.zeros((T, S), dtype=int)

        log_trans = np.log(self.transitions + 1e-10)

        start_global = allowable_states[0]
        scores[0, 0] = scaled_emissions[0, start_global]
        duration[0, 0] = 1

        for t in range(1, T):
            for s in range(S):
                curr_global = allowable_states[s]
                emission = scaled_emissions[t, curr_global]

                score_stay = -np.inf
                if scores[t - 1, s] > -np.inf:
                    score_stay = scores[t - 1, s] + log_trans[curr_global, 0] + emission

                score_move = -np.inf
                if s > 0 and scores[t - 1, s - 1] > -np.inf:
                    prev_global = allowable_states[s - 1]
                    if duration[t - 1, s - 1] >= self.min_state_duration:
                        score_move = scores[t - 1, s - 1] + log_trans[prev_global, 1] + emission

                if score_stay >= score_move:
                    scores[t, s] = score_stay
                    backpointers[t, s] = 0
                    duration[t, s] = duration[t - 1, s] + 1
                else:
                    scores[t, s] = score_move
                    backpointers[t, s] = 1
                    duration[t, s] = 1

        best_final_score = -np.inf
        curr_s = -1

        for s in range(S - 1, -1, -1):
            if scores[T - 1, s] > best_final_score:
                best_final_score = scores[T - 1, s]
                curr_s = s

        if curr_s == -1 or best_final_score == -np.inf:
            return self._proportional_alignment(text, T)

        path = np.zeros(T, dtype=int)
        for t in range(T - 1, -1, -1):
            if curr_s < 0:
                curr_s = 0
            global_id = allowable_states[curr_s]
            path[t] = global_id
            self.prior_counts[global_id] += 1

            if t > 0:
                if backpointers[t, curr_s] == 0:
                    self.trans_counts[global_id, 0] += 1
                else:
                    prev_global = allowable_states[curr_s - 1]
                    self.trans_counts[prev_global, 1] += 1
                    curr_s -= 1

        return path

    def _proportional_alignment(self, text, total_frames):
        state_sequence = []
        for char in text:
            if char in CHAR_TO_STATE_RANGES:
                start, count = CHAR_TO_STATE_RANGES[char]
                for i in range(count):
                    state_sequence.append(start + i)

        if len(state_sequence) == 0:
            space_start, _ = CHAR_TO_STATE_RANGES[' ']
            return np.full(total_frames, space_start, dtype=int)

        indices = np.linspace(0, len(state_sequence) - 1, total_frames).astype(int)
        path = np.array([state_sequence[i] for i in indices], dtype=int)

        for i, state in enumerate(path):
            self.prior_counts[state] += 1
            if i > 0 and path[i] == path[i-1]:
                self.trans_counts[state, 0] += 1
            elif i > 0:
                self.trans_counts[path[i-1], 1] += 1

        return path
