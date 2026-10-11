import torch
import torch.nn as nn

from train_toy import make_data, build_model


device = "cuda"

# torch.manual_seed(42)

# X_train, y_train, _, _ = make_data(42)

# X = X_train[:32].to(device)
# y = y_train[:32].to(device)

# model = build_model("id").to(device)

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=0.01
# )

# criterion = nn.MSELoss()

# scaler = torch.amp.GradScaler("cuda")


# print("===== Before Forward =====")

# print("X dtype               :", X.dtype)
# print("model weight dtype    :", model[0].weight.dtype)
# print("scaler scale          :", scaler.get_scale())


# optimizer.zero_grad()


# # ============================================================
# # Autocast Forward
# # ============================================================

# with torch.autocast(
#     device_type="cuda",
#     dtype=torch.float16
# ):

#     h = model[0](X)

#     print("\n===== Inside Autocast =====")
#     print("Linear output dtype   :", h.dtype)

#     h = model[1](h)
#     h = model[2](h)
#     pred = model[3](h)
#     print("prediction dtype      :", pred.dtype)

#     loss = criterion(
#         pred,
#         y
#     )

#     print("loss dtype            :", loss.dtype)
#     print("loss                  :", loss.item())


# # ============================================================
# # Scaled Backward
# # ============================================================

# scaled_loss = scaler.scale(loss)

# print("\n===== Loss Scaling =====")

# print("original loss         :", loss.item())
# print("scaled loss           :", scaled_loss.item())
# print("scale factor          :", scaler.get_scale())


# scaled_loss.backward()


# grad_before = model[0].weight.grad.norm().item()

# print("\n===== Before Unscale =====")
# print("gradient norm         :", grad_before)


# # ============================================================
# # Unscale Gradient
# # ============================================================

# scaler.unscale_(optimizer)

# print("\n===== All Gradients After Unscale =====")

# for name, p in model.named_parameters():

#     if p.grad is None:
#         print(
#             f"{name:20s} | grad=None"
#         )

#     else:
#         finite = torch.isfinite(
#             p.grad
#         ).all().item()

#         norm = p.grad.norm().item()

#         max_abs = p.grad.abs().max().item()

#         print(
#             f"{name:20s} | "
#             f"norm={norm:.6f} | "
#             f"max={max_abs:.6f} | "
#             f"finite={finite}"
#         )

# grad_after = model[0].weight.grad.norm().item()

# print("\n===== After Unscale =====")
# print("gradient norm         :", grad_after)

# print(
#     "ratio                :",
#     grad_before / grad_after
# )


# # ============================================================
# # Parameter Update
# # ============================================================

# before = model[0].weight.detach().clone()

# scaler.step(optimizer)

# scaler.update()

# after = model[0].weight.detach().clone()

# print("\n===== Optimizer Step =====")

# print(
#     "parameter update norm:",
#     (after - before).norm().item()
# )

# print(
#     "new scaler scale     :",
#     scaler.get_scale()
# )

def dynamic_scale_demo():

    torch.manual_seed(42)

    X_train, y_train, _, _ = make_data(42)

    X = X_train[:32].cuda()
    y = y_train[:32].cuda()

    model = build_model("id").cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    criterion = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", init_scale=65536, backoff_factor=0.5, growth_interval=20)

    for step in range(100):

        optimizer.zero_grad()

        old_scale = scaler.get_scale()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            pred = model(X)

            assert pred.shape == y.shape

            loss = criterion(pred, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        all_finite = True

        for p in model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    all_finite = False

        before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
        }

        scaler.step(optimizer)
        scaler.update()

        new_scale = scaler.get_scale()

        total_change = 0.0

        for name, p in model.named_parameters():
            diff = p.detach() - before[name]
            total_change += diff.pow(2).sum().item()

        total_change = total_change ** 0.5

        print(
            f"step={step + 1} | "
            f"loss={loss.item():.6f} | "
            f"finite={all_finite} | "
            f"scale={old_scale:.0f}->{new_scale:.0f} | "
            f"update_norm={total_change:.6f}"
        )

if __name__ == "__main__":
    dynamic_scale_demo()