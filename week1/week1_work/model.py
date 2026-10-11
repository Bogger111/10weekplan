import torch
import torch.nn as nn

def build_model(norm = "id"):

    if norm == 'id':
        norm_layer = nn.Identity()

    elif norm == "bn":
        norm_layer = nn.BatchNorm1d(32)

    elif norm == "ln":
        norm_layer = nn.LayerNorm(32)

    else :raise ValueError(f"Incorrect Input {norm}")
    
    return nn.Sequential(
        nn.Linear(8, 32),
        norm_layer,
        nn.ReLU(),
        nn.Linear(32, 1)
    )