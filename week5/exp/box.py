import torch
import sys
from pathlib import Path
import numpy as np
import cv2

REPO = Path("/home/bogger/research-repos/4D-Humans")

sys.path.insert(0, str(REPO))

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset

base = Path('/home/bogger/10weekplan/week5/results/d1_baseline_20261026_170644/capture_000.npz')

with np.load(base) as data:
    print(data.files)
    box_center = data['box_center']
    box_size = data['box_size']
    print(box_center.shape, box_size.shape)
    print(box_center, box_size)
    cx, cy = box_center[0]
    s = float(box_size[0])

    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)

    bbox_shape = cfg.MODEL.get("BBOX_SHAPE", None)

    if bbox_shape is None:
        w = h = s
    else:
        target_w, target_h = bbox_shape

        w = s * target_w / max(target_w, target_h)
        h = s * target_h / max(target_w, target_h)

    # 中心点 + 宽高 → 左上角、右下角
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = np.array([
        [x1, y1, x2, y2]
    ], dtype=np.float32)

    print("Boxes:", boxes)
    print("Shape:", boxes.shape)


    # 读取遮挡后的图片
    img_path = '/home/bogger/10weekplan/week5/results/d1_baseline_20261026_170644/input/source.jpg'
    img = cv2.imread(img_path)

    assert img is not None, "Image not found"

    # 调用官方数据集
    dataset = ViTDetDataset(cfg, img, boxes)

    # 获取第一个人的 Crop
    sample = dataset[0]


    print("Baseline Center:", box_center[0])
    print("Current Center:", sample["box_center"])

    print("Baseline Size:", box_size[0])
    print("Current Size:", sample["box_size"])

    print("Image Shape:", sample["img"].shape)

    assert np.allclose(
        sample["box_center"], box_center[0],
        atol=1e-3, rtol=0
    )

    assert np.isclose(
        sample["box_size"], box_size[0],
        atol=1e-3, rtol=0
    )

    print("Fixed BBox verification passed!")

import json

with open(base.parent/'manifest.json') as f:
    manifest = json.load(f)

original_img = cv2.imread(manifest['source_image'])
assert original_img is not None

original_sample = ViTDetDataset(cfg, original_img, boxes)[0]

crop_diff = np.abs(
    sample["img"].astype(np.float32)
    - original_sample["img"].astype(np.float32)
)

print("Crop max difference:", crop_diff.max())
print("Changed elements:", np.count_nonzero(crop_diff > 1e-6))



from hmr2.utils import recursive_to

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))
batch = recursive_to(batch, device)

with torch.inference_mode():
    out = model(batch)

print("Input:", batch["img"].shape)
print("Vertices:", out["pred_vertices"].shape)
print("Camera:", out["pred_cam"].shape)
print("SMPL Parameters:", list(out["pred_smpl_params"].keys()))


save_dict = {
    "pred_vertices": out["pred_vertices"].cpu().numpy(),
    "pred_cam": out["pred_cam"].cpu().numpy(),
    "box_center": batch["box_center"].cpu().numpy(),
    "box_size": batch["box_size"].cpu().numpy(),
}

for key, value in out["pred_smpl_params"].items():
    save_dict[f"smpl_{key}"] = value.cpu().numpy()


save_path = Path(
    "/home/bogger/10weekplan/week5/results/d1_original_fixed_bbox.npz"
)

np.savez_compressed(save_path, **save_dict)

print("Saved:", save_path)




