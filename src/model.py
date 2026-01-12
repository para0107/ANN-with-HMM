import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, feature_dim=60, window_width=9, num_chars=78, states_per_char=3, num_classes=None):
        super(ANN, self).__init__()

        self.input_size = feature_dim * window_width

        if num_classes is not None:
            self.output_size = num_classes
        else:
            self.output_size = num_chars * states_per_char

        self.network = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(128, self.output_size),
            nn.LogSoftmax(dim=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
            batch_size, time_steps, features = x.size()
            x = x.view(-1, features)
            x = self.network(x)
            x = x.view(batch_size, time_steps, -1)
        else:
            x = self.network(x)
        return x
