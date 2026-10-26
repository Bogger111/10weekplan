import numpy as np
import torch
import pathlib

# PATH = pathlib.Path('/home/bogger/10weekplan/week5/results/d1_baseline_20261026_170644/capture_000.npz')

# with np.load(PATH) as data:
#     # print(data.files)
#     # pred_cam = data['pred_cam']
#     # pred_cam_t = data['pred_cam_t']
#     # focal_length = data['focal_length']

#     # print(pred_cam.shape, pred_cam_t.shape, focal_length.shape)

#     # print(pred_cam[0], pred_cam_t[0], focal_length[0])
    
#     # s = pred_cam[0, 0]
#     # f = focal_length[0, 0]

#     # t_z = 2.0 * f / (256.0 * s)

#     # print("Calculated:", t_z)
#     # print("HMR2 output:", pred_cam_t[0, 2])
#     # print("Difference:", abs(t_z - pred_cam_t[0, 2]))

#     points_3d = torch.from_numpy(data['pred_keypoints_3d'])
#     points_2d = torch.from_numpy(data['pred_keypoints_2d'])

#     pred_cam_t = data['pred_cam_t']
#     focal_length = data['focal_length']
#     fx = focal_length[0, 0]
#     fy = focal_length[0, 1]
    
#     my_points_2d = []
#     for i in range(points_3d.shape[1]):
#         u = fx / 256 * (points_3d[0, i, 0] + pred_cam_t[0, 0]) / (points_3d[0, i, 2] + pred_cam_t[0, 2])
#         v = fy / 256 * (points_3d[0, i, 1] + pred_cam_t[0, 1]) / (points_3d[0, i, 2] + pred_cam_t[0, 2])
#         my_points_2d.append(torch.tensor([u, v]))

#     my_points_2d = torch.stack(tuple(my_points_2d), dim=0).unsqueeze(0)

#     print(my_points_2d.shape)
#     loss = torch.linalg.matrix_norm(my_points_2d - points_2d, dim=(-2, -1), keepdim=False).max()

#     print(loss)


def compute_pve(pred: torch.Tensor, gt: torch.Tensor):

    error = pred - gt

    pve = torch.linalg.vector_norm(error, dim=-1).mean()
    print(pve.shape)

    return pve


# gt = torch.zeros(1, 2, 3)
# pred = gt.clone()
# pred[0, 0, 0] = 0.01
# pred[0, 1, 0] = -0.01

# print(compute_pve(pred, gt))

def compute_accel_error(pred, gt, fps=30, frame_ids = None, valid_mask = None):
    a_gt = []
    a_pr = []
    if frame_ids is None:
        frame_ids = torch.arange(gt.shape[0])

    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT shapes must match")

    if gt.shape[0] < 3:
        raise ValueError("At least 3 frames are required")

    new_frame_ids = frame_ids.clone()
    new_frame_ids[~valid_mask] = -1
    
    for i in range(gt.shape[0] - 2):
        
        if new_frame_ids[i+2] - new_frame_ids[i+1] == 1 and new_frame_ids[i+1] - new_frame_ids[i] == 1:        
            a_gt.append((gt[i+2, :, :] - 2*gt[i+1, :, :] + gt[i, :, :])*fps**2)
            a_pr.append((pred[i+2, :, :] - 2*pred[i+1, :, :] + pred[i, :, :])*fps**2)

        else:
            pass

    if len(a_gt) == 0:
        raise ValueError("Not enough continuous frames")

    a_gt = torch.stack(a_gt)
    a_pr = torch.stack(a_pr)    

    error = a_pr - a_gt

    accel_error = torch.linalg.vector_norm(error, dim=-1).mean()

    return accel_error

gt = torch.zeros(6, 1, 3)
pred = gt.clone()

pred[3:, 0, 0] = 1.0

frame_ids = torch.tensor([0, 1, 2, 5, 6, 7])

print(compute_accel_error(
    pred, gt, fps=1, frame_ids=frame_ids
))
