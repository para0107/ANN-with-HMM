#hmm implements what ann cannot do
#1. knowing that some states are more occuring than ohters
#2. knowing that you cannot skip from the start of an 'a' to the end of 'a'
#3. Forced alignment : viterbi algorithm

import numpy as np


class HybridHMM:
    def __init__(self, num_chars=78, states_per_char=7):
        self.total_states = num_chars * states_per_char

        # --- 1. PRIORS (P(q)) ---
        # Initialize uniformly: 1 / 546
        self.priors = np.full(self.total_states, 1.0 / self.total_states)

        # Accumulator to count state occurrences during training
        self.prior_counts = np.zeros(self.total_states)

        # --- 2. TRANSITIONS (A) ---
        # We strictly use Left-to-Right topology.
        # From state 'i', you can only go to 'i' (Self) or 'i+1' (Next).
        # Shape: (Total_States, 2). Col 0 = Self Loop Prob, Col 1 = Next State Prob.
        # Initialize 50/50 chance.
        self.transitions = np.full((self.total_states, 2), 0.5)
        self.trans_counts = np.zeros((self.total_states, 2))

    def reset_accumulators(self):
        """Call this at the start of every Training Epoch."""
        self.prior_counts.fill(0)
        self.trans_counts.fill(0)

    def update_parameters(self):
        """
        The 'M-Step': After aligning all images, update probabilities based on counts.
        """
        # Update Priors
        total_frames = np.sum(self.prior_counts)
        if total_frames > 0:
            self.priors = self.prior_counts / total_frames

        # Update Transitions (Normalize rows so they sum to 1)
        row_sums = np.sum(self.trans_counts, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        self.transitions = self.trans_counts / row_sums

        print("HMM Parameters Updated.")

    def get_scaled_emissions(self, ann_log_output):
        """
        Convert ANN Output P(q|x) -> Scaled Likelihood P(x|q).
        Formula: log(P(x|q)) = log(P(q|x)) - log(P(q))
        """
        # Add tiny epsilon to priors to avoid log(0)
        log_priors = np.log(self.priors + 1e-10)

        # Broadcast subtraction: (Time, States) - (States,)
        return ann_log_output - log_priors

    def forced_alignment(self, scaled_emissions, text_state_indices):
        """
        The Viterbi Algorithm (Forced Mode).
        Finds the best path for the specific text label through the image.

        Args:
            scaled_emissions: Numpy array (T, Total_States) of log-scores.
            text_state_indices: List of state IDs for the text.
                                E.g. "Hi" -> [IDs for H... IDs for i...]
        Returns:
            optimal_path: List of state IDs for each frame (Length T).
        """
        T = scaled_emissions.shape[0]  # Number of time frames in image
        S = len(text_state_indices)  # Number of states in the text sequence

        # The Viterbi Trellis (Grid)
        # We only track the states that appear in our specific text string.
        # scores[t, s] = max score at time t ending in sequence-state s
        scores = np.full((T, S), -np.inf)
        backpointers = np.zeros((T, S), dtype=int)  # 0=Stay, 1=Move

        # Log-Transitions for speed
        log_trans = np.log(self.transitions + 1e-10)

        # --- Initialization (t=0) ---
        # We must start at the very first state of the text
        first_state_global = text_state_indices[0]
        scores[0, 0] = scaled_emissions[0, first_state_global]

        # --- Recursion (Forward Pass) ---
        for t in range(1, T):
            for s in range(S):
                curr_global_id = text_state_indices[s]
                emission = scaled_emissions[t, curr_global_id]

                # 1. Option: Stay in same state (s -> s)
                score_stay = scores[t - 1, s] + log_trans[curr_global_id, 0]

                # 2. Option: Move from previous state (s-1 -> s)
                score_move = -np.inf
                if s > 0:
                    prev_global_id = text_state_indices[s - 1]
                    score_move = scores[t - 1, s - 1] + log_trans[prev_global_id, 1]

                # Winner takes all
                if score_stay > score_move:
                    scores[t, s] = score_stay + emission
                    backpointers[t, s] = 0  # Came from self
                else:
                    scores[t, s] = score_move + emission
                    backpointers[t, s] = 1  # Came from prev

        # --- Backtracking (Backward Pass) ---
        path = np.zeros(T, dtype=int)
        curr_s = S - 1  # Start at the last state of the text

        # If the image was too short for the text, we might end up with -inf.
        # Simple safety: if score is -inf, just perform a linear interpolation (Flat Start)
        if scores[T - 1, curr_s] == -np.inf:
            return None  # Signal failure

        for t in range(T - 1, -1, -1):
            # 1. Record the global state ID in the path
            global_id = text_state_indices[curr_s]
            path[t] = global_id

            # 2. Accumulate counts for M-Step
            self.prior_counts[global_id] += 1

            if t > 0:
                direction = backpointers[t, curr_s]
                if direction == 0:  # Stayed
                    self.trans_counts[global_id, 0] += 1
                elif direction == 1:  # Moved
                    # The transition came FROM the previous state
                    prev_global_id = text_state_indices[curr_s - 1]
                    self.trans_counts[prev_global_id, 1] += 1
                    curr_s -= 1  # Move sequence index back

        return path