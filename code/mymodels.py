import torch.nn as nn


class MyMLP(nn.Module):
  def __init__(self):
    super(MyMLP, self).__init__()
    self.hidden = nn.Linear(70, 16)
    self.out = nn.Linear(16, 5)
    self.ReLU = nn.LeakyReLU()

  def forward(self, x):
    x = self.ReLU(self.hidden(x))
    x = self.out(x)
    return x


class MyCNN(nn.Module):
  def __init__(self):
    super(MyCNN, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv1d(6, 12, 5)
    self.fc1 = nn.Linear(in_features=12*14, out_features=80)
    self.fc2 = nn.Linear(128, 5)
    self.RELU = nn.ReLU()

  def forward(self, x):
    x = self.pool(self.RELU(self.conv1(x)))
    x = self.pool(self.RELU(self.conv2(x)))
    x = x.view(-1, 12*14)
    x = self.RELU(self.fc1(x))
    x = self.fc2(x)
    return x