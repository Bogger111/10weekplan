import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.transformer import(
    sinusoidal_pe,
    SpatialBlock,
    TemporalBlock
)

def generate_skeleton_data(
        num_sequences,
        seed,
        T=9,
        noise_std = 0.03
):
    torch.manual_seed(seed)

    p = torch.tensor([
    [ 0.0,  0.0, 0.0],
    [-0.5,  0.6, 0.0],
    [ 0.5,  0.6, 0.0],
    [-0.3, -0.8, 0.0],
    [ 0.3, -0.8, 0.0],
])

    J = p.shape[0]

    N = num_sequences

    a = (torch.rand(N, 1, 1, 3) - 0.5) * 3 / 5

    v = (torch.rand(N, 1, 1, 3) - 0.5) * 3 / 10
    
    t = torch.linspace(-1, 1, T).reshape(1, T, 1, 1)

    p = p.view(1, 1, J, 3)

    Y = p + a + v * t

    coords = Y.clone()

    noise = torch.randn_like(Y) * noise_std
    
    coords += noise

    center = T // 2

    coords[:, center, :, :] = 0.0

    flag = torch.ones(
        N, T, J, 1,
        dtype=torch.float32
    )

    flag[:, center, :, :] = 0.0

    X = torch.cat([coords, flag], dim=-1)

    return X, Y  

class Skeleton_toy(nn.Module):
    def __init__(self, T = 9, J = 5, C = 32, H = 4):
        super().__init__()
        self.T = T
        self.J = J
        self.C = C
        self.H = H

        self.Spatial = SpatialBlock(C, H)
        self.Temporal = TemporalBlock(C, H)

        Tpe = sinusoidal_pe(T, C).reshape(1, T, 1, C)
        Jpe = sinusoidal_pe(J, C).reshape(1, 1, J, C)

        self.register_buffer('Tpe', Tpe)
        self.register_buffer('Jpe', Jpe)

        self.ffn1 = nn.Linear(4, C)
        self.ffn2 = nn.Linear(C, 3)

        self.norm = nn.LayerNorm(C)

    def forward(self, x):
        x = self.ffn1(x)

        x = x + self.Tpe + self.Jpe

        x, As = self.Spatial(x)

        x, At = self.Temporal(x)

        x = self.norm(x)

        x = self.ffn2(x)

        return x, As, At

def test():
    X, Y = generate_skeleton_data(64, 42)

    N, T, J, D = X.shape

    Net = Skeleton_toy(T, J)

    Y_pred, As, At = Net(X)

    center = T // 2

    loss = torch.nn.functional.mse_loss(Y_pred[:, center], Y[:, center])



    print("pred:", Y_pred.shape)
    print("As:", As.shape)
    print("At:", At.shape)
    print("loss:", loss.item())


    loss.backward()

    print(Net.ffn1.weight.grad is not None)

    print(
        Net.Spatial.Block.mha.wq.weight.grad is not None
    )

    print(
        Net.Temporal.Block.mha.wq.weight.grad is not None
    )

    print(Net.ffn2.weight.grad is not None)

def train():
    X, Y = generate_skeleton_data(8, 42)
    EPOCH = 100
    LR = 1e-3
    N, T, J, D = X.shape
    CENTER = T // 2

    torch.manual_seed(42)

    Net = Skeleton_toy()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=Net.parameters(), lr = LR)
    sclaer = torch.amp.GradScaler('cuda')

    before = Net.ffn2.weight.detach().clone()

    Net.train()
    for step in range(EPOCH):
        optimizer.zero_grad()

        with torch.autocast('cuda'):
            pred, _, _ = Net(X)

            loss = criterion(pred[:, CENTER], Y[:, CENTER])

        sclaer.scale(loss).backward()

        sclaer.unscale_(optimizer)

        sclaer.step(optimizer)

        sclaer.update()

        if (step + 1) % 20 == 0:
            print(f"{step + 1}:, loss{loss.item()}")

            print(
                "spatial grad norm:",
                Net.Spatial.Block.mha.wq.weight.grad.norm().item()
            )

            print(
                "temporal grad norm:",
                Net.Temporal.Block.mha.wq.weight.grad.norm().item()
            )

    change = (
        Net.ffn2.weight.detach() - before
    ).abs().max()

    print("parameter max change:", change.item())

def formal():
    X, Y = generate_skeleton_data(64, 42)
    X_eval, Y_eval = generate_skeleton_data(16, 43)
    
    TrainSet = TensorDataset(X, Y)
    EvalSet = TensorDataset(X_eval, Y_eval)

    TrainLoader = DataLoader(TrainSet, batch_size=8, shuffle=False, pin_memory=True)
    EvalLoader = DataLoader(EvalSet, batch_size=8, shuffle=False, pin_memory=True)

    LR = 1e-3
    CENTER = 4

    torch.manual_seed(42)
    model = Skeleton_toy()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    step = 0

    model.train()
    while step < 300:
        for x, y in TrainLoader:
            optimizer.zero_grad()
            
            pred, _, _ = model(x)

            loss = criterion(pred[:, CENTER], y[:, CENTER])

            loss.backward()

            optimizer.step()

            if (step + 1) % 30 == 0:
                print(f"STEP{step+1}, LOSS{loss.item():6f}")

            step += 1

            if step >= 300:
                break
            

    model.eval()

    total_loss = 0.0
    total_num = 0

    with torch.no_grad():
        for xb, yb in EvalLoader:

            pred, _, _ = model(xb)

            loss = criterion(
                pred[:, CENTER],
                yb[:, CENTER]
            )

            total_loss += loss.item() * xb.shape[0]
            total_num += xb.shape[0]

    val_mse = total_loss / total_num

    print("val mse:", val_mse)

    return model, X_eval, Y_eval

def dependency_test(model, X_eval):
    CENTER = X_eval.shape[1] // 2

    X_test = X_eval[:1].clone().detach()
    X_test.requires_grad_(True)

    model.eval()
    model.zero_grad()

    pred, _, _ = model(X_test)

    scalar = pred[0, CENTER, 0, 0]


    scalar.backward()

    grad = X_test.grad[0, :, :, :3]

    frame_grad = grad.abs().sum(dim=(1, 2))

    print("\n=== Cross-frame dependency ===")

    for t in range(len(frame_grad)):
        mark = "  <- center" if t == CENTER else ""
        print(
            f"frame {t}: "
            f"{frame_grad[t].item():.8f}{mark}"
        )

    visible_grad = torch.cat([
        frame_grad[:CENTER],
        frame_grad[CENTER + 1:]
    ])

    print(
        "\nvisible frame total grad:",
        visible_grad.sum().item()
    )

    print(
        "visible frame max grad:",
        visible_grad.max().item()
    )

import matplotlib.pyplot as plt


def attention_heatmap(model, X_eval):
    model.eval()

    SAMPLE = 0
    HEAD = 0
    JOINT = 0

    CENTER = X_eval.shape[1] // 2

    joint_names = [
        "root",
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot"
    ]

    with torch.no_grad():
        _, As, At = model(X_eval[SAMPLE:SAMPLE+1])

    print("As:", As.shape)
    print("At:", At.shape)

    # =========================================
    # 1. Spatial Attention
    # As: [B, T, H, J, J]
    # =========================================

    spatial = As[
        0, CENTER, HEAD
    ].detach().cpu()

    plt.figure(figsize=(6, 5))

    plt.imshow(spatial)

    plt.colorbar(label="Attention Weight")

    plt.xticks(
        range(len(joint_names)),
        joint_names,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(joint_names)),
        joint_names
    )

    plt.xlabel("Key Joint")
    plt.ylabel("Query Joint")

    plt.title(
        f"Spatial Attention "
        f"(frame={CENTER}, head={HEAD})"
    )

    plt.tight_layout()
    plt.savefig(
        "spatial_attention.png",
        dpi=200
    )

    plt.show()


    # =========================================
    # 2. Temporal Attention
    # At: [B, J, H, T, T]
    # =========================================

    temporal = At[
        0, JOINT, HEAD
    ].detach().cpu()

    T = temporal.shape[0]

    plt.figure(figsize=(6, 5))

    plt.imshow(temporal)

    plt.colorbar(label="Attention Weight")

    plt.xticks(range(T))
    plt.yticks(range(T))

    plt.xlabel("Key Frame")
    plt.ylabel("Query Frame")

    plt.title(
        f"Temporal Attention "
        f"(joint={joint_names[JOINT]}, head={HEAD})"
    )

    plt.tight_layout()
    plt.savefig(
        "temporal_attention.png",
        dpi=200
    )

    plt.show()


    # =========================================
    # 额外打印 center query 的 temporal attention
    # =========================================

    center_attention = temporal[CENTER]

    print("\nCenter-frame query attention:")
    for t, value in enumerate(center_attention):
        print(
            f"frame {t}: "
            f"{value.item():.6f}"
        )