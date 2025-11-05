import torch
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=2):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x expected shape: (batch_size, input_dim)
        return self.fc2(self.relu(self.fc1(x)))

def load_meta_model(path="maml_meta_model.pth", device=torch.device("cpu"), input_dim=384, hidden_dim=128, output_dim=2):
    """
    Loads the meta model state dict saved from Colab into the same architecture.
    """
    model = IntentClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
