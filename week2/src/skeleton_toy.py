import torch

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
        num_sequences, T, J, 1,
        dtype=torch.float32
    )

    flag[:, center, :, :] = 0.0

    X = torch.cat([coords, flag], dim=-1)

    return X, Y  

X, Y = generate_skeleton_data(64, 42)

print(X.shape)
print(Y.shape)

print("masked center:")
print(X[0, 4, :, :3])

print("center flag:")
print(X[0, 4, :, 3])

print("clean target:")
print(Y[0, 4])