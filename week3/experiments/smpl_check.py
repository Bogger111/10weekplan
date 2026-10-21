import torch
import smplx

from pathlib import Path


DTYPE = torch.float32

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "smpl_clean"
    / "SMPL_NEUTRAL.pkl"
)


def exp1_load_smpl():

    print("\n===== Exp 1: Load SMPL =====")

    print("Model path:", MODEL_PATH)
    assert MODEL_PATH.exists(), "SMPL model file not found"

    model = smplx.create(
        model_path=str(MODEL_PATH),
        model_type="smpl",
        gender="neutral",
        num_betas=10,
    )

    model.eval()

    with torch.no_grad():
        output = model()

    print("Vertices:", output.vertices.shape)
    print("Joints:", output.joints.shape)
    print("Faces:", model.faces.shape)

    assert output.vertices.shape == (1, 6890, 3)
    assert torch.isfinite(output.vertices).all()

    print("PASS")


def run_all():
    exp1_load_smpl()


if __name__ == "__main__":
    run_all()