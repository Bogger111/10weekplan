import torch

def make_transform(R:torch.Tensor, t:torch.Tensor):
    assert R.shape[-1] == 3 and R.shape[-2] == 3, "Incorrect Input R"
    assert t.shape[-1] == 3, "Incorrect Input t"

    t = t.unsqueeze(-1)

    zeros = torch.zeros(*R.shape[:-2], 1, 3, device=R.device, dtype=R.dtype)
    ones = torch.ones(*t.shape[:-2], 1, 1, device=t.device, dtype=t.dtype)
    top = torch.cat((R, t), dim=-1)
    bottom = torch.cat((zeros, ones), dim=-1)

    return torch.cat((top, bottom), dim=-2)

def inverse_transform(T:torch.Tensor):
    R = T[..., :3, :3]
    t = T[..., :3, 3:]

    Rt = R.transpose(-1, -2)
    tt = -1 * Rt @ t

    zeros = torch.zeros(*R.shape[:-2], 1, 3, device=R.device, dtype=R.dtype)
    ones = torch.ones(*t.shape[:-2], 1, 1, device=t.device, dtype=t.dtype)
    top = torch.cat((Rt, tt), dim=-1)
    bottom = torch.cat((zeros, ones), dim=-1)   

    return torch.cat((top, bottom), dim=-2)

def compose_transform(T2:torch.Tensor, T1:torch.Tensor):
    return T2 @ T1

def transform_points(T:torch.Tensor, p:torch.Tensor):
    ones = torch.ones(*p.shape[:-1], 1, device=p.device, dtype=p.dtype)
    p = torch.cat((p, ones), dim=-1)
    p = p.unsqueeze(-1)

    out = T @ p

    return out[..., :3, 0]

def forward_kinematics_reference(
    R_local,
    offsets,
    parents,
    root_trans
):
    J = R_local.shape[-3]

    cache = {}

    def get_global(j):

        if j in cache:
            return cache[j]

        if j == 0:
            T_global = make_transform(
                R_local[..., 0, :, :],
                root_trans
            )

        else:
            p = int(parents[j])

            if p < 0 or p >= J:
                raise ValueError(
                    f"Invalid parent {p} for joint {j}"
                )

            T_parent = get_global(p)

            offset_j = offsets[j].to(device=R_local.device, dtype=R_local.dtype)
            offset_j = offset_j.expand(*R_local.shape[:-3], 3)

            T_local = make_transform(
                R_local[..., j, :, :],
                offset_j
            )

            T_global = T_parent @ T_local

        cache[j] = T_global
        return T_global

    R_globals = []
    P_globals = []

    for j in range(J):
        T = get_global(j)

        R_globals.append(
            T[..., :3, :3]
        )

        P_globals.append(
            T[..., :3, 3]
        )

    R_global = torch.stack(
        R_globals,
        dim=-3
    )

    P_global = torch.stack(
        P_globals,
        dim=-2
    )

    return R_global, P_global

def forward_kinematics(
    R_local,
    offsets,
    parents,
    root_trans
):

    J = R_local.shape[-3]

    if offsets.shape != (J, 3):
        raise ValueError(
            f"Expected offsets [{J}, 3], got {offsets.shape}"
        )

    if len(parents) != J:
        raise ValueError(
            f"Expected {J} parents, got {len(parents)}"
        )

    if int(parents[0]) != -1:
        raise ValueError("Root parent must be -1")

    R_global = torch.empty_like(R_local)

    P_global = torch.empty(
        *R_local.shape[:-3],
        J,
        3,
        dtype=R_local.dtype,
        device=R_local.device
    )

    R_global[..., 0, :, :] = R_local[..., 0, :, :]
    P_global[..., 0, :] = root_trans

    for j in range(1, J):

        p = int(parents[j])

        if p < 0 or p >= j:
            raise ValueError(
                f"Invalid parent {p} for joint {j}. "
                "Expected topological order."
            )

        offset_world = (
            R_global[..., p, :, :]
            @ offsets[j].to(
                device=R_local.device,
                dtype=R_local.dtype
            ).unsqueeze(-1)
        ).squeeze(-1)

        P_global[..., j, :] = (
            P_global[..., p, :]
            + offset_world
        )

        R_global[..., j, :, :] = (
            R_global[..., p, :, :]
            @ R_local[..., j, :, :]
        )

    return R_global, P_global