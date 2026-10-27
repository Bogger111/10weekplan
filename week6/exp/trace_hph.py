import torch
from multihmr2 import init_hmr_session, infer_image

session = init_hmr_session(
    '/home/bogger/research-repos/multi-hmr2/checkpoints/multihmr2.pt'
)

decoder = session.model.full_body_decoder
hph = decoder.decoder

records = {}

def hook_hph(module, args, kwargs, output):

    queries = kwargs["x"]
    context = kwargs["context"]

    records["queries_shape"] = tuple(queries.shape)
    records["context_shape"] = tuple(context.shape)
    records["output_shape"] = tuple(output.shape)

    print("\n===== HPH Forward Hook =====")

    print("Queries shape:", queries.shape)
    print("Context shape:", context.shape)
    print("HPH output shape:", output.shape)

    print("\n===== Tensor Properties =====")
    print("Queries dtype:", queries.dtype)
    print("Queries device:", queries.device)
    print("Context dtype:", context.dtype)
    print("Context device:", context.device)

handle = hph.register_forward_hook(
    hook_hph,
    with_kwargs=True
)

block = hph.transformer.layers[0]

self_attn = block[0].fn
cross_attn = block[1].fn

def hook_qkv(module, inputs, output):
    print("Input shape:", inputs[0].shape)
    print("Output shape:", output.shape)

handle_self_qkv = self_attn.to_qkv.register_forward_hook(
    hook_qkv
)

handle_cross_q = cross_attn.to_q.register_forward_hook(
    hook_qkv
)

handle_cross_kv = cross_attn.to_kv.register_forward_hook(
    hook_qkv
)

try:
    pred = infer_image(
        session,
        "/home/bogger/research-repos/multi-hmr2/demo_data/sample_image.jpg",
        conf_thresh=0.4
    )
finally:
    handle.remove()
    handle_cross_kv.remove()
    handle_cross_q.remove()
    handle_self_qkv.remove()


print("\n===== Final Prediction =====")
print("Detected persons:", len(pred))
print("J3D shape:", pred.persons.j3d.shape)
print("K shape:", pred.K.shape)


assert records["queries_shape"][1] == 100
assert records["queries_shape"] == records["output_shape"]
assert records["queries_shape"][0] == records["context_shape"][0]

print("\nHPH Shape Check: PASS")

