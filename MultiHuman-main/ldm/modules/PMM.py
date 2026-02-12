import torch
from torch import nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torch.nn.functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

def prepare_mask(mask, h, w):

    N = mask.shape[0]
    patch_h, patch_w = mask.shape[1]//(64//h), mask.shape[2]//(64//w)
    rescaled_mask = torch.zeros(N, patch_h, patch_w)
    

    rescaled_mask = F.avg_pool2d(mask.unsqueeze(1), 
                                kernel_size=(64//h, 64//w), 
                                stride=(64//h, 64//w)).squeeze(1)
    rescaled_mask = (rescaled_mask > 0).float()
    
    patch_weights = rescaled_mask.view(N, -1)  
    

    for i in range(N):
        max_val = patch_weights[i].max()
        if max_val > 0:
            patch_weights[i] = patch_weights[i] / max_val

        patch_weights[i] = patch_weights[i] * 0.98 + 0.02
    
    return patch_weights


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, pose_bias=None):

        x = self.norm(x)
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, N, N]

        if pose_bias is not None:

            pose_bias_i = pose_bias.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            pose_bias_j = pose_bias.unsqueeze(1).unsqueeze(-2)  # [B, 1, 1, N]
            pose_attention_bias = self.beta * (pose_bias_i + pose_bias_j)  # [B, 1, N, N]
            dots = dots + pose_attention_bias 
        
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, pose_bias):
        for i, (attn, ff) in enumerate(self.layers):
            if pose_bias is not None:
                x_attn = attn(x, pose_bias[i])
                x = x_attn + x
            else:
                x_attn = attn(x, pose_bias)
                x = x_attn + x
            x_ff = ff(x)
            x = x_ff + x
        return x


class PMM(nn.Module):

    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim=512, channels = 320, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.depth = depth

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 128 is the reduced channels
        patch_dim = 128 * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, stride=1, padding=1),
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.h = image_height // patch_height
        self.w = image_width // patch_width
        
        self.pos_embedding = posemb_sincos_2d(
            h = self.h,
            w = self.w,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        

        self.pose_weight_mlp = nn.Sequential(
            nn.Linear(1, dim // 8),  
            nn.GELU(),  
            nn.Dropout(0.1),  
            nn.Linear(dim // 8, dim // 16),
            nn.GELU(),
            nn.Linear(dim // 16, 1),
            nn.Sigmoid()
        )
        
        self.to_out = nn.Sequential(
            Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = patch_height, p2 = patch_width, h = image_size//patch_size),
            nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, img, mask=None):
        device = img.device
        x = self.to_patch_embedding(img)  # 1 320 64 64 ---  1 1024 512
        x = x + self.pos_embedding.to(device, dtype=x.dtype) 

        pose_bias = None
        if mask is not None:
            if mask.shape[0] != x.shape[0]:
                masks = torch.split(mask, x.shape[0])
                raw_pose_bias = [prepare_mask(m, self.h, self.w).to(device) for m in masks]

                pose_bias = []
                for bias in raw_pose_bias:
                    enhanced_bias = self.pose_weight_mlp(bias.unsqueeze(-1)).squeeze(-1)
                    pose_bias.append(enhanced_bias)
                assert self.depth == len(pose_bias)
            else:
                raw_pose_bias = prepare_mask(mask, self.h, self.w).to(device)
                enhanced_bias = self.pose_weight_mlp(raw_pose_bias.unsqueeze(-1)).squeeze(-1)
                pose_bias = [enhanced_bias for _ in range(self.depth)]
            
        x = self.transformer(x, pose_bias)
        x = self.to_out(x)
        
        return x