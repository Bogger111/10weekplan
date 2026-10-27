from pathlib import Path
import torch
import pickle

path = (
    Path.home()
    / "10weekplan/week6/results/d1_offical_default"
    / "anny_params/sample_image.pkl"
)

with open(path, 'rb') as f:
    data = pickle.load(f)

K = data['K']
shape = data['shape']
poses = data['pose_parameters']

def inspector_tensor(name, tensor:torch.Tensor):
    print('shape', tensor.shape, 'dtype', tensor.dtype)
    finite_mask = torch.isfinite(tensor)

    print("All finite:", finite_mask.all().item())
    print("Invalid count:", (~finite_mask).sum().item())


def validate_prediction(K, shape, poses):
    assert K.shape == (3, 3)

    assert shape.shape[-1] == 11

    assert shape.shape[-2] == poses.shape[-4]

    check_pose = poses[..., 3:, :]

    gt_pose = torch.zeros_like(check_pose)

    gt_pose[..., :, 3] = 1

    assert torch.allclose(check_pose, gt_pose)

    

def run_test(name, K, shape, poses, should_pass=True):
    try:
        validate_prediction(K, shape, poses)
        actual_pass = True
    except AssertionError:
        actual_pass = False

    if actual_pass == should_pass:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")


# Test 1: 正常多人
run_test("Normal 13 Persons", K, shape, poses)


# Test 2: 单人
run_test(
    "Single Person",
    K,
    shape[:1],
    poses[:1]
)


# Test 3: 空输出
run_test(
    "Empty Persons",
    K,
    shape[:0],
    poses[:0]
)


# Test 4: 人数不匹配（应该失败）
run_test(
    "Mismatched Persons",
    K,
    shape[:2],
    poses[:3],
    should_pass=False
)


# Test 5: 错误的 K Shape（应该失败）
K_bad = torch.zeros(2, 3, 3)

run_test(
    "Invalid K Shape",
    K_bad,
    shape,
    poses,
    should_pass=False
)


# Test 6: 错误的齐次矩阵（应该失败）
poses_bad = poses.clone()
poses_bad[0, 0, 3, 0] = 1.0

run_test(
    "Invalid Homogeneous Matrix",
    K,
    shape,
    poses_bad,
    should_pass=False
)