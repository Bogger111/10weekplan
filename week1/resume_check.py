import argparse

import torch
import torch.nn as nn

from train_toy import make_data, build_model, train_step


CONFIG = {
    "seed": 42,
    "lr": 0.01,
    "batch_size": 32,
    "total_steps": 100,
    "norm": "id"
}


# ============================================================
# Data
# ============================================================

def get_batch(
    X,
    y,
    step,
    batch_size=CONFIG["batch_size"]
):
    start = (step * batch_size) % len(X)
    end = start + batch_size

    X_batch = X[start:end]
    y_batch = y[start:end]

    return X_batch, y_batch


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    global_step,
    config
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),

        "global_step": global_step,

        "config": config,

        "rng": torch.get_rng_state()
    }

    torch.save(checkpoint, path)

    print(f"checkpoint saved to: {path}")


def load_checkpoint(
    path,
    model,
    optimizer
):
    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    torch.set_rng_state(
        checkpoint["rng"]
    )

    global_step = checkpoint["global_step"]
    config = checkpoint["config"]

    return global_step, config


# ============================================================
# Training
# ============================================================

def train_range(
    X,
    y,
    model,
    optimizer,
    criterion,
    start_step,
    end_step
):
    """
    start_step inclusive
    end_step   exclusive

    train_range(..., 0, 50)
    -> step 0 ~ 49
    -> 共 50 次 optimizer update

    train_range(..., 50, 100)
    -> step 50 ~ 99
    -> 再做 50 次 update
    """

    for step in range(start_step, end_step):

        X_batch, y_batch = get_batch(
            X,
            y,
            step
        )

        loss, grad_norm = train_step(
            model,
            optimizer,
            criterion,
            X_batch,
            y_batch
        )

        if (step + 1) % 20 == 0:
            print(
                f"step={step + 1:3d} "
                f"loss={loss:.6f} "
                f"grad_norm={grad_norm:.6f}"
            )

    return loss


# ============================================================
# Setup
# ============================================================

def make_setup():
    X_train, y_train, _, _ = make_data(
        CONFIG["seed"]
    )

    model = build_model(
        CONFIG["norm"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"]
    )

    criterion = nn.MSELoss()

    return (
        X_train,
        y_train,
        model,
        optimizer,
        criterion
    )


# ============================================================
# Experiment A
# Continuous training: 0 -> 100
# ============================================================

def continuous():

    X, y, model, optimizer, criterion = make_setup()

    print("\n===== Continuous Training =====")

    loss = train_range(
        X,
        y,
        model,
        optimizer,
        criterion,
        0,
        CONFIG["total_steps"]
    )

    torch.save(
        model.state_dict(),
        "continuous_final.pt"
    )

    print(
        "\ncontinuous done",
        f"\nfinal loss = {loss:.6f}"
    )


# ============================================================
# Experiment B - Part 1
# Train 0 -> 50 and save checkpoint
# ============================================================

def save_half():

    X, y, model, optimizer, criterion = make_setup()

    print("\n===== Train First 50 Steps =====")

    loss = train_range(
        X,
        y,
        model,
        optimizer,
        criterion,
        0,
        50
    )

    save_checkpoint(
        path="step50.pt",
        model=model,
        optimizer=optimizer,
        global_step=50,
        config=CONFIG
    )

    print(
        "\nfirst half done",
        "\nglobal_step = 50",
        f"\nloss = {loss:.6f}"
    )


# ============================================================
# Experiment B - Part 2
# Full resume: restore model + optimizer + step + RNG
# ============================================================

def resume_full():

    X, y, model, optimizer, criterion = make_setup()

    print("\n===== Full Resume =====")

    global_step, saved_config = load_checkpoint(
        "step50.pt",
        model,
        optimizer
    )

    print(
        f"loaded checkpoint at global_step={global_step}"
    )

    print(
        "optimizer states =",
        len(optimizer.state_dict()["state"])
    )

    loss = train_range(
        X,
        y,
        model,
        optimizer,
        criterion,
        global_step,
        CONFIG["total_steps"]
    )

    torch.save(
        model.state_dict(),
        "resumed_final.pt"
    )

    print(
        "\nfull resume done",
        f"\nfinal loss = {loss:.6f}"
    )


# ============================================================
# Experiment C
# Model-only resume
# ============================================================

def resume_model_only():

    X, y, model, optimizer, criterion = make_setup()

    print("\n===== Model-only Resume =====")

    checkpoint = torch.load(
        "step50.pt",
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    global_step = checkpoint["global_step"]

    print(
        f"loaded model at global_step={global_step}"
    )

    print(
        "optimizer states BEFORE training =",
        len(optimizer.state_dict()["state"])
    )

    loss = train_range(
        X,
        y,
        model,
        optimizer,
        criterion,
        global_step,
        CONFIG["total_steps"]
    )

    torch.save(
        model.state_dict(),
        "model_only_final.pt"
    )

    print(
        "\nmodel-only resume done",
        f"\nfinal loss = {loss:.6f}"
    )


# ============================================================
# Compare Models
# ============================================================

def max_abs_diff(
    state_a,
    state_b
):

    max_diff = 0.0

    for key in state_a:

        tensor_a = state_a[key]
        tensor_b = state_b[key]

        diff = (
            tensor_a - tensor_b
        ).abs().max().item()

        max_diff = max(
            max_diff,
            diff
        )

    return max_diff


def compare():

    print("\n===== Compare =====")

    continuous_state = torch.load(
        "continuous_final.pt",
        map_location="cpu"
    )

    resumed_state = torch.load(
        "resumed_final.pt",
        map_location="cpu"
    )

    model_only_state = torch.load(
        "model_only_final.pt",
        map_location="cpu"
    )

    full_resume_diff = max_abs_diff(
        continuous_state,
        resumed_state
    )

    model_only_diff = max_abs_diff(
        continuous_state,
        model_only_state
    )

    print(
        "continuous vs full resume :",
        full_resume_diff
    )

    print(
        "continuous vs model-only  :",
        model_only_diff
    )

    print()

    if full_resume_diff <= 1e-6:
        print("FULL RESUME: PASS")
    else:
        print("FULL RESUME: FAIL")

    if model_only_diff > full_resume_diff:
        print("MODEL-ONLY: diverged as expected")
    else:
        print("MODEL-ONLY: unexpected result")

def step51_probe():
    print("\n===== Step 51 Probe =====")

    X, y, model_full, opt_full, criterion = make_setup()
    _, _, model_fresh, opt_fresh, _ = make_setup()

    checkpoint = torch.load(
        "step50.pt",
        map_location="cpu"
    )


    model_full.load_state_dict(
        checkpoint["model"]
    )
    opt_full.load_state_dict(
        checkpoint["optimizer"]
    )


    model_fresh.load_state_dict(
        checkpoint["model"]
    )

    step = checkpoint["global_step"]   

    X_batch, y_batch = get_batch(
        X, y, step
    )


    opt_full.zero_grad()
    opt_fresh.zero_grad()

    pred_full = model_full(X_batch)
    pred_fresh = model_fresh(X_batch)

    loss_full = criterion(
        pred_full, y_batch
    )

    loss_fresh = criterion(
        pred_fresh, y_batch
    )

    print(
        "loss diff =",
        abs(
            loss_full.item()
            - loss_fresh.item()
        )
    )


    loss_full.backward()
    loss_fresh.backward()

    grad_diff = (
        model_full[0].weight.grad
        - model_fresh[0].weight.grad
    ).abs().max().item()

    print(
        "gradient diff =",
        grad_diff
    )


    opt_full.step()
    opt_fresh.step()

    param_diff = (
        model_full[0].weight
        - model_fresh[0].weight
    ).abs().max().item()

    print(
        "parameter diff after step =",
        param_diff
    )

# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Checkpoint / Resume Experiment"
    )

    parser.add_argument(
        "mode",
        choices=[
            "continuous",
            "save_half",
            "resume_full",
            "resume_model_only",
            "compare",
            "step51_probe"
        ],
        help="which experiment to run"
    )

    args = parser.parse_args()

    if args.mode == "continuous":
        continuous()

    elif args.mode == "save_half":
        save_half()

    elif args.mode == "resume_full":
        resume_full()

    elif args.mode == "resume_model_only":
        resume_model_only()

    elif args.mode == "compare":
        compare()

    elif args.mode == "step51_probe":
        step51_probe()


if __name__ == "__main__":
    main()