"""CNN baseline variant augmented with dropout for MNIST experiments."""
import torch.nn as nn
import torch.nn.functional as F


class CNNDropout(nn.Module):
    """Baseline CNN with dropout layers configurable via constructor."""

    def __init__(self, dropout_prob: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3, 3))
        self.dropout2d = nn.Dropout2d(p=dropout_prob)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = self.dropout2d(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
