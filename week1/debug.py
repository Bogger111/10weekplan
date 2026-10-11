import argparse

import torch
import torch.nn as nn


# ============================================================
# Gradient Inspector
# ============================================================

def inspect_grad(model):
    print("\n===== Gradient Inspector =====")

    for name, p in model.named_parameters():

        if p.grad is None:
            print(
                f"{name:20s} | grad=None"
            )

        else:
            finite = torch.isfinite(
                p.grad
            ).all().item()

            grad_norm = p.grad.norm().item()

            print(
                f"{name:20s} | "
                f"norm={grad_norm:.6f} | "
                f"finite={finite}"
            )


# ============================================================
# Common Setup
# ============================================================

def make_setup():

    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    X = torch.randn(32, 8)
    y = torch.randn(32, 1)

    criterion = nn.MSELoss()

    return model, X, y, criterion


# ============================================================
# Failure 1
# Accidental detach
# ============================================================

def detach_bug():

    print("\n===== Detach Bug =====")

    model, X, y, criterion = make_setup()

    # 手动 forward
    h = model[0](X)
    h = model[1](h)

    # 故意断开计算图
    h = h.detach()

    pred = model[2](h)

    loss = criterion(
        pred,
        y
    )

    print(
        f"loss = {loss.item():.6f}"
    )

    loss.backward()

    inspect_grad(model)


# ============================================================
# Failure 1 Fixed
# ============================================================

def detach_fixed():

    print("\n===== Detach Fixed =====")

    model, X, y, criterion = make_setup()

    h = model[0](X)
    h = model[1](h)

    # 不 detach
    pred = model[2](h)

    loss = criterion(
        pred,
        y
    )

    print(
        f"loss = {loss.item():.6f}"
    )

    loss.backward()

    inspect_grad(model)


# ============================================================
# Failure 2
# Non-finite loss / gradient
# ============================================================

def nonfinite_bug():

    print("\n===== Non-finite Bug =====")

    model, X, y, criterion = make_setup()

    pred = model(X)

    mse = criterion(
        pred,
        y
    )

    # 故意制造非法 sqrt
    #
    # pred.abs().mean() >= 0
    # 所以负号后通常 < 0
    bad_value = -pred.abs().mean()

    bad_term = torch.sqrt(
        bad_value
    )

    loss = mse + bad_term

    print(
        f"mse = {mse.item():.6f}"
    )

    print(
        "bad_value =",
        bad_value.item()
    )

    print(
        "bad_term =",
        bad_term.item()
    )

    print(
        "loss =",
        loss.item()
    )

    print(
        "loss finite =",
        torch.isfinite(loss).item()
    )

    loss.backward()

    inspect_grad(model)


# ============================================================
# Anomaly Detection
# ============================================================

def anomaly_debug():

    print("\n===== Anomaly Detection =====")

    model, X, y, criterion = make_setup()

    try:

        with torch.autograd.detect_anomaly():

            pred = model(X)

            mse = criterion(
                pred,
                y
            )

            bad_term = torch.sqrt(
                -pred.abs().mean()
            )

            loss = mse + bad_term

            loss.backward()

    except RuntimeError as e:

        print("\nCaught RuntimeError:")
        print(e)

        print(
            "\nAnomaly detection successfully "
            "located the problematic backward op."
        )


# ============================================================
# Failure 2 Fixed
# ============================================================

def nonfinite_fixed():

    print("\n===== Non-finite Fixed =====")

    model, X, y, criterion = make_setup()

    pred = model(X)

    mse = criterion(
        pred,
        y
    )

    # 合法输入：
    # pred.abs().mean() >= 0
    #
    # +1e-8 避免 sqrt 输入过于接近 0
    safe_value = (
        pred.abs().mean()
        + 1e-8
    )

    good_term = torch.sqrt(
        safe_value
    )

    loss = mse + good_term

    print(
        f"mse = {mse.item():.6f}"
    )

    print(
        "safe_value =",
        safe_value.item()
    )

    print(
        "good_term =",
        good_term.item()
    )

    print(
        "loss =",
        loss.item()
    )

    print(
        "loss finite =",
        torch.isfinite(loss).item()
    )

    loss.backward()

    inspect_grad(model)


# ============================================================
# Run Everything
# ============================================================

def run_all():

    detach_bug()

    detach_fixed()

    nonfinite_bug()

    anomaly_debug()

    nonfinite_fixed()


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Gradient Debugging Experiments"
    )

    parser.add_argument(
        "mode",
        choices=[
            "detach_bug",
            "detach_fixed",
            "nonfinite_bug",
            "anomaly",
            "nonfinite_fixed",
            "all"
        ],
        help="which debugging experiment to run"
    )

    args = parser.parse_args()

    if args.mode == "detach_bug":
        detach_bug()

    elif args.mode == "detach_fixed":
        detach_fixed()

    elif args.mode == "nonfinite_bug":
        nonfinite_bug()

    elif args.mode == "anomaly":
        anomaly_debug()

    elif args.mode == "nonfinite_fixed":
        nonfinite_fixed()

    elif args.mode == "all":
        run_all()


if __name__ == "__main__":
    main()