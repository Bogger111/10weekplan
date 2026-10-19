import torch
from src.kinematics import inverse_transform

def my_linear_blend_skinning(
    vertice: torch.Tensor,
    weights,
    G_pose,
    G_rest
):
    """
    vertices: [V, 3]
    weights:  [V, J]
    G_pose:   [B, T, J, 4, 4]
    G_rest:   [J, 4, 4]

    return:
        posed_vertices: [B, T, V, 3]
    """
    A = torch.matmul(G_pose, inverse_transform(G_rest)) #[B, T, J, 4, 4]
    ones = torch.ones(*vertice.shape[:-1], 1, dtype=vertice.dtype, device=vertice.device)
    Vertice = torch.cat((vertice, ones), dim=1) #[V, 4]
    Vertice = Vertice.unsqueeze(-1) #[V, 4, 1]  
    posed_list = []
    for i in range(Vertice.shape[0]):
        V = torch.matmul(A, Vertice[i])
        
        w_i = weights[i].view(1, 1, -1, 1, 1)
        
        v_blended = torch.sum(w_i * V, dim=2).squeeze(-1)
        
        posed_list.append(v_blended[..., :3])
        
    posed_vertices = torch.stack(posed_list, dim=2)
    return posed_vertices

def linear_blend_skinning(
    vertice: torch.Tensor,
    weights,
    G_pose,
    G_rest        
):
    A = torch.matmul(G_pose, inverse_transform(G_rest)) #[B, T, J, 4, 4]
    ones = torch.ones(*vertice.shape[:-1], 1, dtype=vertice.dtype, device=vertice.device)
    Vertice = torch.cat((vertice, ones), dim=1) #[V, 4]
    Vertice = Vertice.unsqueeze(-1) #[V, 4, 1] 
    T = torch.einsum('vj, btjkl -> btvkl', weights, A) #[B, T, V, 4, 4]
    posed_vertices =torch.matmul(T, Vertice) #[B, T, V, 4, 1]
    return posed_vertices[..., :3, 0]


