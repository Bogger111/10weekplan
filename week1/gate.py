import time

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)


DEVICE = "cuda"
SEED = 42
TOTAL_STEPS = 100


# ============================================================
# Seed
# ============================================================


torch.manual_seed(SEED)


# ============================================================
# Data
# ============================================================

X = torch.randn(256, 8)

true_w = torch.randn(8, 1)

# [256]
y = X @ true_w + 0.1 * torch.randn(256, 1)

dataset = TensorDataset(
    X,
    y,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
)

X_eval = torch.randn(32, 8)

w = torch.randn(8, 1)

y_eval = X_eval @ true_w + 0.1 * torch.randn(32, 1)



# ============================================================
# Model
# ============================================================

model = nn.Sequential(
    nn.Linear(8, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(DEVICE)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)

criterion = nn.MSELoss()

scaler = torch.amp.GradScaler(
    "cuda"
)


# ============================================================
# Resume
# ============================================================

start_step = 0

try:
    checkpoint = torch.load(
        "checkpoint.pt"
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint['optimizer']
    )

    scaler.load_state_dict(
        checkpoint['scaler']
    )

    torch.set_rng_state(checkpoint['rng'])

    start_step = checkpoint["global_step"]

except FileNotFoundError:
    pass


# ============================================================
# Evaluation
# ============================================================

def evaluate():

    model.eval()

    with torch.no_grad():
        X_eval = X_eval.to(DEVICE)
        y_eval = y_eval.to(DEVICE)

        pred = model(X_eval)

        loss = criterion(
            pred,
            y_eval,
        )

    return loss.item()


# ============================================================
# Training
# ============================================================

model.train()

iterator = iter(loader)

optimizer.zero_grad()

torch.cuda.empty_cache()
torch.cuda.synchronize()

start_time = time.perf_counter()


for step in range(
    start_step,
    TOTAL_STEPS
):

    try:
        X_batch, y_batch = next(iterator)

    except StopIteration:
        iterator = iter(loader)
        X_batch, y_batch = next(iterator)

    X_batch = X_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):

        pred = model(X_batch)

        loss = criterion(
            pred,
            y_batch,
        )


    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    optimizer.zero_grad()

    scaler.scale(
        loss
    ).backward()

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )




    scaler.step(
        optimizer
    )

    scaler.update()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (step + 1) % 20 == 0:

        val_loss = evaluate()
        model.train()

        print(
            f"step={step + 1} "
            f"train={loss.item():.6f} "
            f"val={val_loss:.6f}"
        )


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if (step + 1) == 50:

        torch.save(
            {
                "model": model.state_dict(),
                "global_step": step + 1,
                'optimizer': optimizer.state_dict(),
                'rng': torch.get_rng_state(),
                'scaler': scaler.state_dict()
            },
            "checkpoint.pt",
        )

torch.cuda.synchronize()

elapsed = (
    time.perf_counter()
    - start_time
)

print(
    "training time:",
    elapsed
)