from pathlib import Path
import pickle
import torch
import anny

model = anny.create_fullbody_model(
    local_changes=True,
    pose_parameterization='root_relative_world',
    remove_unattached_vertices=False
).to(dtype=torch.float32, device='cpu')

model.eval()

print('Number of bones:', model.bone_count)
print('Phenityoes:', model.phenotype_labels)

original_dir = Path(
    "/home/bogger/10weekplan/week6/results/"
    "blender_baseline_20261029_203921/video/anny_params"
)

smoothed_dir = Path(
    "/home/bogger/10weekplan/week6/results/anny_smoothed"
)

file = sorted(original_dir.glob("*.pkl"))[0]

with open(file, "rb") as f:
    original = pickle.load(f)

with open(smoothed_dir / file.name, "rb") as f:
    smoothed = pickle.load(f)


def rebuild_mesh(data):
    pose = data["pose_parameters"].float()
    shape = data["shape"].float()

    assert pose.shape[1] == model.bone_count
    assert shape.shape[1] == len(model.phenotype_labels)

    phenotype_kwargs = {
        label: shape[:, i]
        for i, label in enumerate(model.phenotype_labels)
    }

    local_changes_kwargs = {
        key: torch.zeros(shape.shape[0], dtype=shape.dtype)
        for key in model.local_change_labels
    }

    with torch.inference_mode():
        output = model(
            pose_parameters=pose,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs,
        )

    return output["vertices"]


vertices_before = rebuild_mesh(original)
vertices_after = rebuild_mesh(smoothed)

print(vertices_before.shape)
print(vertices_after.shape)