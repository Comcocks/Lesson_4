import torch
import torch.nn as nn
import torch.nn.functional as F

class FCModel(nn.Module):
    def __init__(self, input_size=784, num_classes=10, hidden_sizes=[256, 128, 64]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0]),
                       nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                       nn.Linear(hidden_sizes[1], hidden_sizes[2])])
        self.dropout = nn.Dropout(0.25)
        self.fc_out = nn.Linear(hidden_sizes[2], num_classes)

    def forward(self, x):
        x = self.flatten(x)
        for l in self.layers:
            x = F.relu(l(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class FCModelCIFAR(nn.Module):
    def __init__(self, input_size=3072, num_classes=10, hidden_sizes=[1024, 512, 256, 128]):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0]),
                       nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                       nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                       nn.Linear(hidden_sizes[2], hidden_sizes[3])])
        self.dropout = nn.Dropout(0.4)
        self.fc_out = nn.Linear(hidden_sizes[3], num_classes)

    def forward(self, x):
        x = self.flatten(x)
        for l in self.layers:
            x = F.relu(l(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x