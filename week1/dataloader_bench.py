import time
import statistics

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)

from train_toy import (
    make_data,
    build_model,
)


DEVICE = "cuda"

BATCH_SIZE = 32
EPOCHS = 100
REPEATS = 3


# ============================================================
# Dataset stays on CPU
# ============================================================

X_train, y_train, _, _ = make_data(42)

dataset = TensorDataset(
    X_train,
    y_train,
)


def run_once(num_workers):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(
            num_workers > 0
        ),
        pin_memory=True
    )

    model = build_model().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    criterion = nn.MSELoss()

    model.train()

    total_samples = 0

    torch.cuda.synchronize()

    start = time.perf_counter()

    for epoch in range(EPOCHS):

        for X, y in loader:

            # CPU → GPU
            X = X.to(DEVICE,non_blocking = True)
            y = y.to(DEVICE, non_blocking = True)

            optimizer.zero_grad()

            pred = model(X)

            loss = criterion(
                pred,
                y,
            )

            loss.backward()

            optimizer.step()

            total_samples += len(X)

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    throughput = (
        total_samples / elapsed
    )

    return elapsed, throughput


def benchmark(num_workers):

    times = []
    throughputs = []

    for repeat in range(REPEATS):

        elapsed, throughput = run_once(
            num_workers
        )

        times.append(elapsed)
        throughputs.append(throughput)

        print(
            f"workers={num_workers} | "
            f"repeat={repeat + 1} | "
            f"time={elapsed:.4f}s | "
            f"throughput={throughput:.1f} samples/s"
        )

    print()

    print(
        f"workers={num_workers} median time = "
        f"{statistics.median(times):.4f}s"
    )

    print(
        f"workers={num_workers} median throughput = "
        f"{statistics.median(throughputs):.1f} samples/s"
    )

    print(
        f"time range = "
        f"{min(times):.4f} ~ {max(times):.4f}s"
    )

    print()

    return {
        "workers": num_workers,
        "time": statistics.median(times),
        "throughput": statistics.median(
            throughputs
        ),
    }


if __name__ == "__main__":

    result_0 = benchmark(
        num_workers=0
    )

    result_2 = benchmark(
        num_workers=2
    )

    print(
        "===== Comparison ====="
    )

    print(
        f"workers=0 | "
        f"time={result_0['time']:.4f}s | "
        f"throughput={result_0['throughput']:.1f}"
    )

    print(
        f"workers=2 | "
        f"time={result_2['time']:.4f}s | "
        f"throughput={result_2['throughput']:.1f}"
    )

    print(
        "speedup workers2/workers0 =",
        result_0["time"]
        / result_2["time"]
    )