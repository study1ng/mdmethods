from torch import autocast, nn
import torch
from mamba_ssm import Mamba

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # Bi-directional scan
        self.mambas = nn.ModuleList(
            [Mamba(dim, d_state, d_conv, expand) for _ in range(2)]
        )

        self.out_proj = nn.Linear(dim * 2, dim)

        self.channel_token = channel_token  ## whether to use channel as tokens

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # Bi-directional scan
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x_mamba = self.mambas[0](x_norm)
        x_flip = x_norm.flip([1]).contiguous()
        x_flip_mamba = self.mambas[1](x_flip).flip([1]).contiguous()

        x_mamba = torch.cat([x_mamba, x_flip_mamba], dim=-1)
        x_mamba = self.out_proj(x_mamba)

        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        # Bi-directional scan
        x_norm = self.norm(x_flat)

        x_mamba = self.mambas[0](x_norm)
        x_flip = x_norm.flip([1]).contiguous()
        x_flip_mamba = self.mambas[1](x_flip).flip([1]).contiguous()

        x_mamba = torch.cat([x_mamba, x_flip_mamba], dim=-1)
        x_mamba = self.out_proj(x_mamba)

        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    @autocast(device_type="cuda", enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out
