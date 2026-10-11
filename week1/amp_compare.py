import copy
import time

import torch
import torch.nn as nn

from train_toy import make_data, build_model

SEED = 42
DEVICE = 'cuda'
STEPS = 1000
BATCH_SIZE = 32
LR = 0.01

X_train, y_train, _, _ = make_data(SEED)

X_train = X_train.to(DEVICE)
y_train = y_train.to(DEVICE)

base_model = build_model().to(DEVICE)

initial_state = copy.deepcopy(
    base_model.state_dict()
)

criterion = nn.MSELoss()

def get_batch(step):
    start = (step * BATCH_SIZE) % len(X_train)
    end = start + BATCH_SIZE

    X_batch = X_train[start: end]
    y_batch = y_train[start: end]

    return (X_batch, y_batch)

def evaluate(model):
    model.eval()

    with torch.no_grad():
        pred = model(X_train)

        loss = criterion(pred, y_train)

    return loss.item()

def run_FP32():
    model = build_model().to(DEVICE)

    model.load_state_dict(initial_state)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    for step in range(STEPS):
        X, y = get_batch(step)

        optimizer.zero_grad()

        pred = model(X)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

    torch.cuda.synchronize()
    elapsed = (
        time.perf_counter() - start_time
    )

    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**2
    )

    final_loss = evaluate(model)

    return {
        "mode": "FP32",
        "loss": final_loss,
        "time": elapsed,
        "memory": peak_memory
    }

def run_amp():
    model = build_model().to(DEVICE)

    model.load_state_dict(initial_state)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    scaler = torch.amp.GradScaler('cuda',init_scale=32768)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    for step in range(STEPS):
        X, y = get_batch(step)

        optimizer.zero_grad()

        with torch.autocast(
            device_type='cuda',
            dtype=torch.float16
        ):
            pred = model(X)

            loss = criterion(pred, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        scaler.step(optimizer)

        scaler.update()

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time)

    peak_memory = (torch.cuda.max_memory_allocated() / 1024**2)

    final_loss = evaluate(model)

    return {
        "mode": "AMP",
        "loss": final_loss,
        "time": elapsed,
        "memory": peak_memory,
        "scale": scaler.get_scale()
    }

if __name__ == "__main__":

    fp32 = run_FP32()
    amp = run_amp()

    print("\n===== FP32 vs AMP =====")

    print(
        f"FP32 | "
        f"loss={fp32['loss']:.6f} | "
        f"time={fp32['time']:.4f}s | "
        f"peak_memory={fp32['memory']:.2f} MB"
    )

    print(
        f"AMP  | "
        f"loss={amp['loss']:.6f} | "
        f"time={amp['time']:.4f}s | "
        f"peak_memory={amp['memory']:.2f} MB | "
        f"final_scale={amp['scale']:.0f}"
    )

    print()

    print(
        "speed ratio FP32/AMP =",
        fp32["time"] / amp["time"]
    )

    print(
        "memory ratio AMP/FP32 =",
        amp["memory"] / fp32["memory"]
    )