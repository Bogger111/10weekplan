import torch

def reprojection_loss(uv_pred, uv_gt, valid, image_size=(640, 480)):
    W, H = image_size
    size = torch.sum(valid, dim=-1)
    if size == 0:
        raise ValueError('Incorrect Input.')
    u_pred = torch.where(valid[...,None], uv_pred[..., :1], 0)
    u_gt = torch.where(valid[...,None], uv_gt[..., :1], 0)

    v_pred = torch.where(valid[...,None], uv_pred[..., 1:], 0)
    v_gt = torch.where(valid[...,None], uv_gt[..., 1:], 0)

    uloss = torch.pow((u_pred - u_gt) / W, exponent=2) #[B, N ,1]
    vloss = torch.pow((v_pred - v_gt) / H, exponent=2) #[B, N, 1]
    loss = torch.sum((uloss + vloss), (-1, -2)) / size #[B]

    return loss