from pathlib import Path
import pickle
import torch
import anny

model = anny.create_fullbody_model(
    local_changes=True,
    pose_parameterization='root_relative_world',
    remove_unattached_vertices=False,
    all_phenotypes=True
).to(dtype=torch.float32, device='cpu')

model.eval()

print('Number of bones:', model.bone_count)
print("Phenontypes:", model.phenotype_labels)


original_dir = Path(
    '/home/bogger/10weekplan/week6/results/blender_baseline_20261029_203921/video/anny_params'
)

smoothed_dir = Path(
    '/home/bogger/10weekplan/week6/results/anny_smoothed'
)

file = sorted(original_dir.glob('*.pkl'))[0]

with open(file, 'rb') as f:
    original = pickle.load(f)

with open(smoothed_dir / file.name, 'rb') as f:
    smoothed = pickle.load(f)


def rebuild_mesh(data):
    pose = data['pose_parameters'].to(
        dtype=torch.float32,
        device="cpu"
    )

    shape = data['shape'].to(
        dtype=torch.float32,
        device="cpu"
)
    assert pose.shape[1] == model.bone_count
    assert shape.shape[1] == len(model.phenotype_labels)

    phenotype_kwargs = {
        label: shape[:, i]
        for i, label in enumerate(model.phenotype_labels)
    }

    local_changes_kwargs = {
        key: torch.zeros_like(shape[:, 0], dtype=shape.dtype)
        for key in model.local_change_labels
    }

    with torch.inference_mode():
        output = model(
            pose_parameters =pose,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_changes_kwargs
        )

    return output['vertices']


def check():
    person_id = 0

    root_delta = (
        smoothed["pose_parameters"][person_id, 0, :3, 3]
        - original["pose_parameters"][person_id, 0, :3, 3]
    )

    vertex_delta = (
        vertices_after[person_id]
        - vertices_before[person_id]
    )

    error = torch.linalg.vector_norm(
        vertex_delta - root_delta,
        dim=-1
    )

    print("Root translation:", root_delta)
    print("Mean vertex error:", error.mean().item())
    print("Max vertex error:", error.max().item())  


def find_max_error():
    max_displacement = 0
    best_file = None

    for file in sorted(original_dir.glob("*.pkl")):
        with open(file, "rb") as f:
            before = pickle.load(f)

        with open(smoothed_dir / file.name, "rb") as f:
            after = pickle.load(f)

        delta = (
            after["pose_parameters"][:, 0, :3, 3]
            - before["pose_parameters"][:, 0, :3, 3]
        )

        displacement = torch.linalg.vector_norm(
            delta, dim=-1
        ).max().item()

        if displacement > max_displacement:
            max_displacement = displacement
            best_file = file

    print("Largest displacement:", max_displacement)
    print("Frame:", best_file.name)   

vertices_before = rebuild_mesh(original)
vertices_after = rebuild_mesh(smoothed)

print(vertices_before.shape)
print(vertices_after.shape)

find_max_error()