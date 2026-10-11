import time

import torch
import torch.nn as nn

from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function
)

from train_toy import make_data, build_model

DEVICE = "cuda"
STEPS = 100

X_train, y_train, _, _ = make_data(42)

X_train = X_train.to(DEVICE)
y_train = y_train.to(DEVICE)

model = build_model().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

criterion = nn.MSELoss()


def get_batch(step):

    batch_size = 32

    start = (
        step * batch_size
    ) % len(X_train)

    end = start + batch_size

    return (
        X_train[start:end],
        y_train[start:end]
    )


with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:

    for step in range(STEPS):

        X, y = get_batch(step)

        with record_function("zero_grad"):
            optimizer.zero_grad()

        with record_function("forward"):
            pred = model(X)
            loss = criterion(pred, y)

        with record_function("backward"):
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()


print(
    prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    )
)