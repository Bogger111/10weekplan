import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
)

from train_toy import make_data, build_model


DEVICE = "cuda"
BATCH_SIZE = 32
PROFILE_STEPS = 100


# ============================================================
# Dataset stays on CPU
# ============================================================

X_train, y_train, _, _ = make_data(42)

dataset = TensorDataset(
    X_train,
    y_train,
)


def run_profile(num_workers):

    print(
        f"\n========== num_workers={num_workers} =========="
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,

        persistent_workers=(
            num_workers > 0
        ),
    )

    model = build_model().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    criterion = nn.MSELoss()

    model.train()

    iterator = iter(loader)

    # ========================================================
    # Warmup
    # 不把第一次 CUDA / worker 初始化算进正式 profile
    # ========================================================

    for _ in range(10):

        try:
            X, y = next(iterator)

        except StopIteration:
            iterator = iter(loader)
            X, y = next(iterator)

        X = X.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad()

        pred = model(X)

        loss = criterion(
            pred,
            y,
        )

        loss.backward()

        optimizer.step()

    torch.cuda.synchronize()

    # ========================================================
    # Profile
    # ========================================================

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:

        for step in range(PROFILE_STEPS):

            # --------------------------------------------
            # DataLoader
            # --------------------------------------------

            with record_function("data_loading"):

                try:
                    X, y = next(iterator)

                except StopIteration:
                    iterator = iter(loader)
                    X, y = next(iterator)

            # --------------------------------------------
            # CPU → GPU
            # --------------------------------------------

            with record_function("h2d_transfer"):

                X = X.to(
                    DEVICE,
                    non_blocking=True,
                )

                y = y.to(
                    DEVICE,
                    non_blocking=True,
                )

            # --------------------------------------------
            # zero_grad
            # --------------------------------------------

            with record_function("zero_grad"):
                optimizer.zero_grad()

            # --------------------------------------------
            # forward
            # --------------------------------------------

            with record_function("forward"):

                pred = model(X)

                loss = criterion(
                    pred,
                    y,
                )

            # --------------------------------------------
            # backward
            # --------------------------------------------

            with record_function("backward"):
                loss.backward()

            # --------------------------------------------
            # optimizer
            # --------------------------------------------

            with record_function("optimizer_step"):
                optimizer.step()

    torch.cuda.synchronize()

    # ========================================================
    # Print
    # ========================================================

    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=25,
        )
    )


if __name__ == "__main__":

    run_profile(
        num_workers=0
    )

    run_profile(
        num_workers=2
    )