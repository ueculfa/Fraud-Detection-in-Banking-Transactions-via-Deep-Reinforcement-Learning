import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(FraudDQN, self).__init__()
        # Giriş katmanı: 11 risk faktörü
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        # Çıkış katmanı: 2 Aksiyon (0: Normal, 1: Şüpheli/Alarm)
        self.output = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)