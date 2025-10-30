"""CNN variant with half the convolutional filters of the baseline model."""
import torch.nn as nn
import torch.nn.functional as F


class CNNHalf(nn.Module):
    """MNIST CNN where each convolution stage uses half the filters of the baseline."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=(3, 3))
        self.fc1 = nn.Linear(in_features=10 * 10 * 10, out_features=250)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
