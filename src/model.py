#The network must look at the sliding window of features and predict which HMM
#state (of the 546 states) is visible(the current feature belongs to)

import torch.nn as nn
class ANN(nn.Module):
    def __init__(self, feature_dim=60, window_width=9, num_chars=78, states_per_char=7):
        """
                The Neural Network architecture defined in the paper.

                Args:
                    feature_dim (int): Dimension of a single feature vector (60).
                    window_width (int): Size of the sliding window (9).
                    num_chars (int): Number of characters in alphabet (78).
                    states_per_char (int): HMM states per character (7).
                """
        super(ANN, self).__init__()

        self.input_size = feature_dim*window_width #60 features * 9 frames = 540 input neurons

        self.output_size = num_chars*states_per_char #78 chars * 7 states  =546 output neurons(classes)

        #Definde layers
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 192),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(192,128),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(128, self.output_size),
            nn.LogSoftmax(dim=1)# log probabilities for the HMM math
        )
        self._init_weights()
    def _init_weights(self):
        #Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (Batch_Size, 540)
        Returns:
            Tensor: Log-probabilities of shape (Batch_Size, 546)
        """
        # Flatten the window input if it comes in as (Batch, 9, 60)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.network(x)