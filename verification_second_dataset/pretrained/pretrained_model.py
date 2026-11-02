"""Pre-trained fraud detector of PHNPR/Credit-Card-Fraud-Detection (MIT licence, see LICENSE_PHNPR.txt),
re-created in PyTorch. Keras model: Dense(20, relu) -> Dense(7, relu) -> Dense(5, relu) -> Dense(1, sigmoid),
29 inputs. Weights are loaded unchanged from NN_model_weights.h5; forward() returns the logit."""
import os, h5py, numpy as np, torch, torch.nn as nn
H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_model_weights.h5")

class PretrainedMLP(nn.Module):
    def __init__(self, n_features=29):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 20), nn.ReLU(), nn.Linear(20, 7), nn.ReLU(),
                                 nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_pretrained():
    m = PretrainedMLP()
    with h5py.File(H5, "r") as f:
        for i, l in enumerate([l for l in m.net if isinstance(l, nn.Linear)]):
            name = "dense" if i == 0 else f"dense_{i}"
            l.weight.data = torch.tensor(np.array(f[name][name]["kernel:0"]).T, dtype=torch.float32)
            l.bias.data = torch.tensor(np.array(f[name][name]["bias:0"]), dtype=torch.float32)
    return m.eval()
