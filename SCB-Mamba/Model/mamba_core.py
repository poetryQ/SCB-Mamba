import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Callable
import numpy as np


class MambaConfig:
    def __init__(self,
                 d_model: int = 512,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_rank: str = "auto",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init: str = "random",
                 dt_scale: float = 1.0,
                 dt_init_floor: float = 1e-4,
                 bias: bool = False,
                 conv_bias: bool = True,
                 use_fast_path: bool = True):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias
        self.use_fast_path = use_fast_path


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
        batch, seqlen, dim = u.shape
        d_state = A.shape[-1]

        if delta_bias is not None:
            delta = delta + delta_bias[None, None, :]

        if delta_softplus:
            delta = F.softplus(delta)

        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(1) * u.unsqueeze(-1)

        x = torch.zeros(batch, dim, d_state, device=u.device, dtype=u.dtype)
        ys = []

        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.sum(x * C[:, i].unsqueeze(1), dim=-1)
            if D is not None:
                y = y + u[:, i] * D
            if z is not None:
                y = y * F.silu(z[:, i])
            ys.append(y.unsqueeze(1))

        y = torch.cat(ys, dim=1)

        ctx.save_for_backward(u, delta, A, B, C, D, z, deltaA, deltaB_u, x, delta_bias)
        ctx.delta_softplus = delta_softplus
        ctx.d_state = d_state

        return y

    @staticmethod
    def backward(ctx, dy):
        u, delta, A, B, C, D, z, deltaA, deltaB_u, x, delta_bias = ctx.saved_tensors
        delta_softplus = ctx.delta_softplus
        d_state = ctx.d_state

        batch, seqlen, dim = u.shape

        if delta_softplus:
            delta = F.softplus(delta)

        du = torch.zeros_like(u)
        ddelta = torch.zeros_like(delta)
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)
        dC = torch.zeros_like(C)
        dD = torch.zeros_like(D) if D is not None else None
        dz = torch.zeros_like(z) if z is not None else None
        ddelta_bias = torch.zeros_like(delta_bias) if delta_bias is not None else None

        dx = torch.zeros(batch, dim, d_state, device=dy.device, dtype=dy.dtype)

        for i in reversed(range(seqlen)):
            y_grad = dy[:, i]

            if z is not None:
                y_grad = y_grad * F.silu(z[:, i])

            dC_i = torch.sum(dx * x, dim=1)
            dC[:, i] += dC_i

            dx = dx + y_grad.unsqueeze(-1) * C[:, i].unsqueeze(1)

            if D is not None:
                du_i = y_grad * D
                du[:, i] += du_i

            if z is not None:
                dz_i = y_grad * (u[:, i] * D if D is not None else 0.0) * F.silu(z[:, i]) * (1 - F.silu(z[:, i]))
                dz[:, i] += dz_i

            dB_i = torch.sum(dx * delta[:, i].unsqueeze(-1) * u[:, i].unsqueeze(-1), dim=0)
            dB += dB_i

            ddelta_i = torch.sum(dx * B.unsqueeze(1) * u[:, i].unsqueeze(-1), dim=(0, 2))
            ddelta[:, i] += ddelta_i

            dA_i = torch.sum(dx * x * delta[:, i].unsqueeze(-1), dim=(0, 1))
            dA += dA_i

            if i > 0:
                dx = deltaA[:, i] * dx

        if delta_bias is not None:
            ddelta_bias = torch.sum(ddelta, dim=(0, 1))

        if delta_softplus:
            ddelta = ddelta * torch.exp(-delta)

        return du, ddelta, dA, dB, dC, dD, dz, ddelta_bias, None


class MambaSSM(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        self.A = nn.Parameter(torch.randn(config.d_inner, config.d_state))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.log(self.A))
        self.D = nn.Parameter(torch.ones(config.d_inner))

        self.dt_proj_weight = nn.Parameter(torch.randn(config.d_inner, config.dt_rank))
        self.dt_proj_bias = nn.Parameter(torch.randn(config.d_inner))

        self.B_proj = nn.Linear(config.d_inner, config.d_state, bias=False)
        self.C_proj = nn.Linear(config.d_inner, config.d_state, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )

        self.act = nn.SiLU()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.A_log, mean=0.0, std=0.02)
        nn.init.normal_(self.D, mean=0.0, std=0.02)
        nn.init.normal_(self.dt_proj_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dt_proj_bias, mean=0.0, std=0.02)

        nn.init.kaiming_normal_(self.B_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.C_proj.weight, nonlinearity='linear')

        nn.init.kaiming_normal_(self.conv1d.weight, nonlinearity='linear')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, dim = x.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        dt = F.linear(x, self.dt_proj_weight, self.dt_proj_bias)
        dt = F.softplus(dt)

        x_dbl = F.linear(x, self.B_proj.weight)
        B = self.B_proj(x_dbl)
        C = self.C_proj(x_dbl)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seqlen]
        x = x.transpose(1, 2)

        x = self.act(x)

        y = SelectiveScan.apply(x, dt, A, B, C, D, None, None, False)

        return y


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        self.ssm = MambaSSM(config)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        self.norm = nn.LayerNorm(config.d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.in_proj.weight, nonlinearity='linear')
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity='linear')
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        x_proj = self.in_proj(x)
        x, z = x_proj.chunk(2, dim=-1)

        x = self.ssm(x)
        x = x * F.silu(z)

        x = self.out_proj(x)

        return x + residual


class MambaEncoder(nn.Module):
    def __init__(self, config: MambaConfig, num_layers: int = 12):
        super().__init__()
        self.config = config
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlappingPatchEmbedding(nn.Module):
    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 patch_size: int = 16,
                 stride: int = 8,
                 in_channels: int = 3,
                 embed_dim: int = 512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)

        output_size = [(img_size[i] + 2 * (patch_size // 2) - patch_size) // stride + 1
                       for i in range(2)]
        self.grid_size = output_size
        self.num_patches = output_size[0] * output_size[1]

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x