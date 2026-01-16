import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from mamba_ssm import selective_scan_fn

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class GSR(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=8, num_layers=2):
        """
        Global Step-size Reweighting (GSR) Model
        """
        super().__init__()
        self.seq_len = seq_len
        # FFN1
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # FFN2
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_len)
        )
        self.proj = nn.Linear(input_dim, 1)
        self.num_layers = num_layers

    def gamma_like_prior(self, seq_len, alpha=2.0, beta=1.0, device='cpu'):
        if not torch.is_tensor(alpha):
            alpha = torch.tensor([alpha])
        if not torch.is_tensor(beta):
            beta = torch.tensor([beta])
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(0)
        if beta.ndim == 1:
            beta = beta.unsqueeze(0)
        B, _ = alpha.shape
        eps = alpha
        x = torch.arange(seq_len, device=device).float().unsqueeze(0) # 1, T
        unnorm = torch.exp(alpha * torch.log(x + eps) - x / beta)
        prior = unnorm / unnorm.sum(-1, keepdim=True)  # normalize to sum 1
        return prior  # shape: [B, T]

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.shape
        for _ in range(self.num_layers):
            x = x + self.token_mlp(x)  # token mixing
            x = x.transpose(1, 2)  # [B, D, T]
            x = x + self.channel_mlp(x)
            x = x.transpose(1, 2)  # back to [B, T, D]

        # random prior distribution
        rand_T = torch.rand(1).to(x) * (T-1) + 1
        alpha = x.new_ones(B, 1) * rand_T
        prior = self.gamma_like_prior(x.shape[1], alpha=alpha, beta=torch.ones_like(alpha), device=x.device)
        if torch.any(torch.isnan(prior)):
            prior = self.gamma_like_prior(x.shape[1], alpha=alpha, beta=torch.ones_like(alpha), device=x.device)
        # reweighting coefficients
        logits = self.proj(x).squeeze(-1)  # [B, T]
        weight = torch.sigmoid(logits)  # [B, T]
        return weight, prior

class GSRMamba(nn.Module):
    def __init__(self, d_model, seq_len, d_state=16, expand=2, dt_rank='auto',
                 d_conv=4, conv_bias=True, bias=False, lambda_kl=0.1,
                 ):
        """
        Global Step-size Reweighted Mamba (GSRMamba)
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.d_conv = d_conv
        self.d_state = d_state
        self.bias = bias
        self.conv_bias = conv_bias
        self.lambda_kl = lambda_kl
        if dt_rank == 'auto':
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.gsr = GSR(
            d_model * expand, seq_len,
            hidden_dim=d_state,
            num_layers=2
        )

        self.norm = RMSNorm(d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x, hidden_state=None, return_att_args=False):
        (b, l, d) = x.shape
        if hidden_state is None:
            conv_state, ssm_state = None, None
        else:
            conv_state, ssm_state = hidden_state
            # cut state
            if type(ssm_state) is list or type(ssm_state) is tuple:
                conv_state, ssm_state = conv_state[:b], tuple([s[:b] for s in ssm_state])
            else:
                conv_state, ssm_state = conv_state[:b], ssm_state[:b]

        x_and_res = self.in_proj(self.norm(x))  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        org_x = x
        if conv_state is None:
            x = self.conv1d(x)[:, :, :l]
        else:
            x = self.conv1d(torch.cat([conv_state, x], dim=-1))[:, :, self.d_conv:l+self.d_conv]
        if self.training:
            if conv_state is None:
                conv_state = org_x.new_zeros(b, self.d_inner, self.d_conv)
            conv_state = torch.cat([conv_state[..., 1:], org_x[:, :, :1]], dim=-1)
        else:
            conv_state = org_x[:, :, -self.d_conv:]

        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)

        # The process of GSR guided S6
        y, ssm_state, kl_loss, args = self.ssm(x, res, return_att_args=return_att_args)

        output = self.out_proj(y)

        return output, (conv_state.detach(), ssm_state.detach()), kl_loss, args

    def kl_loss(self, weight, prior, eps=1e-8):
        # Normalize to probability distributions
        weight = weight / (weight.sum(dim=1, keepdim=True) + eps)  # [B, T]
        prior = prior / (prior.sum(dim=-1, keepdim=True) + eps)  # [T] or [B, T]

        # Add eps to avoid log(0)
        log_weight = torch.log(weight + eps)
        log_prior = torch.log(prior + eps)

        # KL divergence
        kl = (weight * (log_weight - log_prior)).sum(dim=1)  # [B]
        return kl.mean()

    def ssm(self, x, res, return_att_args=False):
        """
        GSR guided S6
        """
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))

        # Global Step-size Reweighting (GSR) model
        weight, prior = self.gsr(x)
        kl_loss = self.kl_loss(weight, prior)
        delta = delta * weight.unsqueeze(-1)

        # SSM process in GSR guided S6
        y, ssm_state = selective_scan_fn(
            x.transpose(-1, -2),
            delta.transpose(-1, -2),
            A,
            B.transpose(-1, -2),
            C.transpose(-1, -2),
            self.D.float(),
            z=res.transpose(-1, -2),
            delta_bias=torch.zeros_like(self.dt_proj.bias),
            delta_softplus=False,
            return_last_state=True,
        )
        y = y.transpose(-1, -2)

        if return_att_args:
            # A-> D, N; dB -> B, T, D, N; C -> B, T, N; d -> B, T, D
            args = (A.detach().cpu(), einsum(delta, B, 'b l d_h, b l n -> b l d_h n').detach().cpu(),
                    C.detach().cpu(), delta.detach().cpu(), prior.detach().cpu())
        else:
            args = ()

        return y, ssm_state, kl_loss, args