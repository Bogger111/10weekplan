import smplx

def load_smpl_model(model_path, device = 'cpu'):
    return smplx.create(model_path, model_type='smpl').to(device)
def smpl_forward(
        model,
        betas,
        body_pose,
        global_orient,
        transl,
):
    output = model(
        betas,
        body_pose,
        global_orient,
        transl
    )
    vertices = output.vertices
    joints = output.joints

    return vertices, joints