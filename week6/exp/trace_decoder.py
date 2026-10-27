import torch
from multihmr2.api import init_hmr_session, infer_image

CONF_THRESH = 0.4
NMS_THRESH = 0.25

session = init_hmr_session(
    "/home/bogger/research-repos/multi-hmr2/checkpoints/multihmr2.pt"
)

decoder = session.model.full_body_decoder
records = {}


def hook_conf(module, inputs, output):
    records["logits"] = output.detach().float().cpu()


handle = decoder.mlp_conf.register_forward_hook(hook_conf)

try:
    pred = infer_image(
        session,
        "/home/bogger/research-repos/multi-hmr2/demo_data/sample_image.jpg",
        conf_thresh=CONF_THRESH,
        dist_thresh_nms=NMS_THRESH
    )
finally:
    handle.remove()


logits = records["logits"]

conf = torch.sigmoid(logits)

Q = (conf > CONF_THRESH).sum().item()

N = len(pred)

print("Logits shape:", logits.shape)
print("Confidence shape:", conf.shape)

print("J3D shape:", pred.persons.j3d.shape)

print("\n===== Decoder Statistics =====")
print("Total queries:", conf.numel())
print("After confidence:", Q)
print("After NMS:", N)
print("Suppressed by NMS:", Q - N)

print("\nFinal confidence shape:", pred.persons.conf.shape)
print("Final confidence:", pred.persons.conf.squeeze(-1))






