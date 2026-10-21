import torch

def mpjpe(pred, gt, mask=None):
    error = torch.linalg.vector_norm(pred - gt, dim=-1)

    if mask is None:
        return error.mean()

    if not mask.any():
        raise ValueError('No valid joints')

    return error[mask].mean()
def root_aligned_mpjpe(pred, gt, root_index=0, mask=None):
    B, J, C = pred.shape

    if mask is not None:
        if mask.shape != (B, J) or mask.dtype != torch.bool:
            raise ValueError("Invalid mask shape or dtype")

        if not mask[:, root_index].all():
            raise ValueError("Root joint is invalid")

    P = pred - pred[:, root_index:root_index+1, :]
    G = gt - gt[:, root_index:root_index+1, :]

    return mpjpe(P, G, mask)


def pa_mpjpe(pred, gt, mask=None):
    if pred.ndim != 3 or pred.shape[-1] != 3 or pred.shape != gt.shape:
        raise ValueError("pred and gt must have shape [B, J, 3]")

    if (pred.dtype != gt.dtype
            or pred.device != gt.device
            or not pred.is_floating_point()):
        raise ValueError("pred and gt must share floating dtype and device")

    B, J, _ = pred.shape

    if mask is None:
        mask = torch.ones(
            B, J, dtype=torch.bool, device=pred.device
        )
    elif (mask.shape != (B, J)
          or mask.dtype != torch.bool
          or mask.device != pred.device):
        raise ValueError("mask must be bool [B, J] on the same device")

    count = mask.sum(dim=1).reshape(B, 1, 1)

    if torch.any(count < 3):
        raise ValueError("Each sample needs at least 3 valid joints")

    if (not torch.isfinite(pred[mask]).all()
            or not torch.isfinite(gt[mask]).all()):
        raise ValueError("Valid joints must be finite")

    valid = mask.unsqueeze(-1)  

    pred_safe = torch.where(valid, pred, torch.zeros_like(pred))
    gt_safe = torch.where(valid, gt, torch.zeros_like(gt))


    muP = pred_safe.sum(dim=1, keepdim=True) / count
    muG = gt_safe.sum(dim=1, keepdim=True) / count

    P = torch.where(valid, pred_safe - muP, torch.zeros_like(pred))
    G = torch.where(valid, gt_safe - muG, torch.zeros_like(gt))


    H = P.transpose(-1, -2) @ G

    U, S, Vh = torch.linalg.svd(H)


    tol = 100 * torch.finfo(pred.dtype).eps
    if torch.any(S[:, 1] <= tol * S[:, 0]):
        raise ValueError("Degenerate or nearly collinear joint set")


    D = torch.eye(
        3, dtype=pred.dtype, device=pred.device
    ).unsqueeze(0).repeat(B, 1, 1)

    det = torch.linalg.det(U @ Vh)

    D[:, 2, 2] = torch.where(
        det < 0,
        -torch.ones_like(det),
        torch.ones_like(det),
    )

    R = U @ D @ Vh


    P_rot = P @ R

    numerator = (P_rot * G).sum(
        dim=(1, 2), keepdim=True
    )
    denominator = (P ** 2).sum(
        dim=(1, 2), keepdim=True
    )

    scale = numerator / denominator

    if torch.any(scale <= 0):
        raise ValueError("Nonpositive fitted scale")


    t = muG - scale * (muP @ R)


    pred_aligned = scale * (pred_safe @ R) + t

    error = torch.linalg.vector_norm(
        pred_aligned - gt_safe, dim=-1
    )

    return error[mask].mean()
