import torch
import torch.nn as nn

class VIMESemiSupervised(nn.Module):
    def __init__(self, predictor_input_dim, predictor_hidden_dim, predictor_output_dim):
        super().__init__()
        self.fc1 = nn.Linear(predictor_input_dim, predictor_hidden_dim)
        self.fc2 = nn.Linear(predictor_hidden_dim, predictor_hidden_dim)
        self.fc3 = nn.Linear(predictor_hidden_dim, predictor_output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
