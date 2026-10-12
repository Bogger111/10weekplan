import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels = None, out_channels = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
    
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device
        )

        random_tensor.floor_()

        return x / keep_prob * random_tensor
    
class Attention(nn.Module):
    def __init__(self, dim, heads, attn_drop = 0., fc_drop = 0.):
        assert dim % heads == 0, "ValueError in dim or heads"

        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads

        self.scale = self.head_dim ** -0.5
        self.wqkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.fc = nn.Linear(dim, dim)
        self.fc_drop = nn.Dropout(fc_drop)

    def forward(self, x):

        B, N, C = x.shape

        qkv = self.wqkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv 

        s = q @ k.transpose(-1, -2) * self.scale

        a = nn.functional.softmax(s, dim=-1)
        a = self.attn_drop(a)

        x = (a @ v).transpose(1 ,2).reshape(B, N, C)
        x = self.fc(x)
        x = self.fc_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, heads, norm_layer = nn.LayerNorm, droppath = 0., drop = 0., attn_drop = 0., mlp_ratio = 2):
        super().__init__()
        self.mlp = MLP(dim, dim*mlp_ratio, dim, drop=drop)
        self.mha = Attention(dim, heads, attn_drop, drop)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.droppath = DropPath(droppath)

    def forward(self, x):
        x = x + self.droppath(self.mha(self.norm1(x)))
        x = x + self.droppath(self.mlp(self.norm2(x)))

        return x

class PoseFormer(nn.Module):
    def __init__(self, T, J, C, dim, depth, heads, norm_layer = nn.LayerNorm, droppath = 0., drop = 0., attn_drop = 0., pos_drop = 0.2, mlp_ratio = 2):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, droppath, depth)]
        
        self.dim = dim

        self.Embedding = nn.Linear(C, dim)
        
        self.SpatialPE = nn.Parameter(torch.zeros(1, J, dim))
        self.TemporalPE = nn.Parameter(torch.zeros(1, T, J*dim))

        self.posdrop = nn.Dropout(pos_drop)

        self.SpatialFormer = nn.ModuleList([
            Block(dim, heads, norm_layer, dpr[i], drop, attn_drop, mlp_ratio)
            for i in range(depth)
            ])
        self.TemporalFormer = nn.ModuleList([
            Block(J*dim, heads, norm_layer, dpr[i], drop, attn_drop, mlp_ratio)
            for i in range(depth)
        ])

        self.Spatialnorm = norm_layer(dim)
        self.Temporalnorm = norm_layer(J*dim)

        self.weigh_average = nn.Conv1d(T, 1, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(J*dim),
            nn.Linear(J*dim, J*3)
        )

    def Spatial_forward(self, x):
        x = self.Embedding(x) + self.SpatialPE
        x = self.posdrop(x)
        for model in self.SpatialFormer:
            x = model(x)
        x = self.Spatialnorm(x)

        return x

    def Temporal_forward(self, x):
        x = x + self.TemporalPE
        x = self.posdrop(x) 
        for model in self.TemporalFormer:
            x = model(x)
        x = self.Temporalnorm(x)

        return x

    def Average_forward(self, x):
        return self.weigh_average(x)

    def Head(self, x):
        return self.head(x)

    def forward(self, x):
        B, T, J, C = x.shape
        dim = self.dim

        x = x.reshape(B*T, J, C)
        x = self.Spatial_forward(x).reshape(B, T, J*dim)
        x = self.Temporal_forward(x)
        x = self.Average_forward(x)
        x = self.Head(x).reshape(B, 1, J, 3)

        return x

def test():
    model = PoseFormer(
        T=9,
        J=17,
        C=2,
        dim=32,
        depth=4,
        heads=8
    )

    x = torch.randn(
        2, 9, 17, 2,
        requires_grad=True
    )

    y = model(x)

    print("output:", y.shape)

    loss = y.pow(2).mean()
    loss.backward()

    print(
        "embedding grad:",
        model.Embedding.weight.grad.norm().item()
    )

    print(
        "spatial grad:",
        model.SpatialFormer[0]
             .mha.wqkv.weight.grad.norm().item()
    )

    print(
        "temporal grad:",
        model.TemporalFormer[0]
             .mha.wqkv.weight.grad.norm().item()
    )

    print(
        "head grad:",
        model.head[1].weight.grad.norm().item()
    )