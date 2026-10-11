import torch
import torch.nn as nn

def make_data(seed = 42):
    torch.manual_seed(seed)

    true_w = torch.randn(8, 1)

    X_train = torch.randn(256, 8)
    y_train = X_train @ true_w + 0.1 * torch.randn(256, 1)

    X_val = torch.randn(64, 8)
    y_val = X_val @ true_w + 0.1 * torch.randn(64, 1)

    return X_train, y_train, X_val, y_val

def build_model(norm = "id"):

    if norm == "id":
        norm_layer = nn.Identity()
    elif norm == "bn":
        norm_layer = nn.BatchNorm1d(32)
    elif norm == "ln":
        norm_layer = nn.LayerNorm(32)
    else:
        raise ValueError(f"unkonow norm{norm}")

    model = nn.Sequential(
        nn.Linear(8, 32),
        norm_layer,
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model


def train_step(model, optimizer, criterion, X, y):

    model.train()

    optimizer.zero_grad()

    pred = model(X)

    assert pred.shape == y.shape, f"{pred.shape} != {y.shape}"

    loss = criterion(pred, y)

    loss.backward()

    grad_norm = model[0].weight.grad.norm().item()

    optimizer.step()

    return loss.item(), grad_norm

def val(model, criterion, X, y):
    model.eval()

    with torch.no_grad():
        pred = model(X)
        assert pred.shape == y.shape
        loss = criterion(pred, y)

    return loss.item()


    

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = make_data()

    model = build_model()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.MSELoss()

    batch_size = 32

    for i in range(100):

        start = (i * batch_size) % len(X_train)
        end = start + batch_size

        X_batch = X_train[start: end]
        y_batch = y_train[start: end]

        loss, grad_norm = train_step(
            model, optimizer, criterion, X_batch, y_batch
            )

        if (i + 1) % 10 == 0:
            print(i+1, loss, grad_norm)

    print(val(model, criterion, X_val, y_val))

