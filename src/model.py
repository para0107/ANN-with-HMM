import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, feature_dim=60, window_width=9, num_chars=78, states_per_char=3, num_classes=None):
        """
        The Neural Network architecture.
        Updated: Uses ReLU and BatchNorm for stable training.
        """
        super(ANN, self).__init__()

        self.input_size = feature_dim * window_width  # 540 input neurons

        if num_classes is not None:
            self.output_size = num_classes
        else:
            self.output_size = num_chars * states_per_char

        # --- IMPROVED ARCHITECTURE ---
        # 1. Replaced Sigmoid with ReLU (prevents vanishing gradients)
        # 2. Added BatchNorm1d (stabilizes learning)
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(self.input_size, 256),  # Increased size slightly
            nn.BatchNorm1d(256),  # New: Normalization
            nn.ReLU(),  # New: Activation
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # New: Normalization
            nn.ReLU(),  # New: Activation
            nn.Dropout(0.3),

            # Output Layer
            nn.Linear(128, self.output_size),
            nn.LogSoftmax(dim=1)
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming (He) Initialization is better for ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch_Size, Time_Steps, 540)
        Returns:
            Tensor: Log-probabilities of shape (Batch_Size, Time_Steps, Num_Classes)
        """
        # 1. Save original dimensions
        if x.dim() == 3:
            batch_size, time_steps, features = x.size()

            # 2. Flatten Batch and Time: (Batch * Time, 540)
            # This treats every frame as an independent sample
            x = x.view(-1, features)

            # 3. Pass through network
            x = self.network(x)  # -> (Batch * Time, Output_Size)

            # 4. Reshape back: (Batch, Time, Output_Size)
            x = x.view(batch_size, time_steps, -1)

        else:
            # Fallback for 2D inputs (if any)
            x = self.network(x)

        return x