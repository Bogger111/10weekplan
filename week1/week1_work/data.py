import torch
from torch.utils.data import DataLoader, TensorDataset

def make_data(seed):
    X_train = torch.randn(256, 8)
    true_w = torch.randn(8, 1)
    y_train = X_train @ true_w + 0.1 * torch.randn(256, 1)

    X_eval = torch.randn(32, 8)
    y_eval = X_eval @ true_w + 0.1 * torch.randn(32, 1)

    def MyDataloader(X, y):
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            pin_memory=True
        )
        return dataloader
    

    return MyDataloader(X_train, y_train), MyDataloader(X_eval, y_eval)

