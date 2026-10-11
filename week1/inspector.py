def inspect_gradients(model):
    print("=" * 80)

    for name, p in model.named_parameters():

        if p.grad is None:
            print(f"{name:20s} | grad=None")
            continue

        g = p.grad

        print(
            f"{name:20s} | "
            f"mean={g.mean().item(): .6f} | "
            f"std={g.std().item(): .6f} | "
            f"max={g.abs().max().item(): .6f} | "
            f"norm={g.norm().item(): .6f}"
        )