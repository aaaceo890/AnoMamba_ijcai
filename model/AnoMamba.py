import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from model_utils.gsrmamba import GSRMamba

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FFNMixing(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.ffn2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return x + self.ffn2(self.act(self.ffn1(x)))

class AnoMamba(nn.Module):
    def __init__(self,
                 input_size,
                 window_size,
                 d_model=16,
                 state_size=8,
                 patch_size=8,
                 patch_stride=4,
                 expand=2,
                 block_num=3,
                 lambda_kl=0.1,
                 ):
        super().__init__()
        self.enc_in = input_size
        self.window_size = window_size
        self.block_num = block_num
        # for kl loss
        self.lambda_kl = lambda_kl

        # for patch
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.revin_layer = RevIN(input_size, affine=True, subtract_last=False)
        # self.revin_layer = None
        self.embedding = TokenEmbedding(self.patch_size, d_model)

        remainder = (self.window_size - self.patch_size) % self.patch_stride
        pad_len = (self.patch_stride - remainder) % self.patch_stride
        num_patches = (self.window_size + pad_len - self.patch_size) // self.patch_stride + 1

        self.blocks = nn.ModuleList([
            GSRMamba(
                d_model,
                num_patches,
                d_state=state_size,
                expand=expand,
                d_conv=4,
                lambda_kl=lambda_kl,
            ) for _ in range(block_num)]
        )

        self.ffn_mixing = FFNMixing(input_size, d_ff=input_size * 2)
        self.head = nn.Linear(d_model * num_patches, window_size)

    def patchify(self, x, patch_size, patch_stride):
        # x -> B, T, D
        B, T, D = x.shape

        remainder = (T - patch_size) % patch_stride
        pad_len = (patch_stride - remainder) % patch_stride
        num_patches = (T + pad_len - patch_size) // patch_stride + 1
        x_pad = torch.cat([x.new_zeros(B, pad_len, D), x], dim=1)
        x_patch = x_pad.unfold(dimension=1, size=patch_size, step=patch_stride)

        x_patch = x_patch.permute(0, 1, 3, 2)
        return x_patch

    def forward(self, x, hidden=None, return_att_args=False):
        B, T, D = x.shape
        if B != 1:
            hidden = None

        if self.revin_layer is not None:
            x = self.revin_layer(x, 'norm')

        # Step 1. Patch Embedding
        # B, N, P, D
        x = self.patchify(x, self.patch_size, self.patch_stride)
        B, N, P, D = x.shape
        # (BD), N, P -> (BD), N, Dm
        x = self.embedding(x.permute(0, 3, 1, 2).reshape(B*D, N, P))
        Dm = x.shape[-1]

        # Step 2. Global Step-size Reweighted Mamba
        kl_losses = []
        all_args = []
        x_s = x
        if hidden is None:
            hidden = [None] * self.block_num
        new_hidden = []
        for i in range(self.block_num):
            x_s, new_h, kl_loss, args = self.blocks[i](x_s, hidden[i], return_att_args=return_att_args)
            kl_losses.append(kl_loss)
            new_hidden.append(new_h)
            all_args.append(args)

        # Step 3. Channel Mixing
        # BD, N, Dm -> B, N, Dm, D
        x_s = x_s.reshape(B, D, N, Dm).permute(0, 2, 3, 1)
        x_s = self.ffn_mixing(x_s)

        # Step 4. Reconstruction output
        # B, D, N, P -> B, D, T
        y = self.head(x_s.permute(0, 3, 1, 2).reshape(B, D, -1))

        rec = y.transpose(1, 2) # B, T, D

        if self.revin_layer is not None:
            rec = self.revin_layer(rec, 'denorm')

        kl_loss = torch.stack(kl_losses).mean()

        res = {
            "x": rec,
            "kl_loss": kl_loss,
            "all_args": all_args,
            "hidden": new_hidden,
        }

        return res

    @property
    def metric_tags(self):
        tags = ["total_loss", "reconstruction_loss", "kl_loss"]
        return tags

    def cal_loss(self, res, target, epoch, **kwargs):
        x, kl_loss, _, _ = res.values()
        rec_loss = nn.functional.mse_loss(target, x)
        loss = rec_loss + self.lambda_kl * kl_loss
        metrics = (loss.item(),)
        count = (1,)
        metrics += (rec_loss.item(),)
        count += (1,)
        metrics += (kl_loss.item(),)
        count += (1,)

        return loss, metrics, count

    def anomaly_detection(self, test_dataloader, device="cpu"):
        self.eval()
        scores = []
        y_trues = []
        y_hats = []
        cat_args = [[] for _ in range(self.block_num)]
        with torch.no_grad():
            for data, target in tqdm(test_dataloader):
                if device != "cpu":
                    data = tuple([d.cuda(0) for d in data]) if type(data) is list else data.cuda(0)
                    target = target.cuda(0)
                x, _, all_args, _ = self.forward(data, return_att_args=False).values()
                for i in range(self.block_num):
                    cat_args[i].append(all_args[i])
                if test_dataloader.dataset.align == "causal_pad":
                    score = nn.functional.mse_loss(target, x[:, -1].unsqueeze(1), reduction="none")
                    y_hats.append(x[:, -1].unsqueeze(1).cpu().detach().numpy().reshape(-1, x.shape[-1]))
                    y_trues.append(target.cpu().detach().numpy().reshape(-1, x.shape[-1]))
                else:
                    score = nn.functional.mse_loss(target, x, reduction="none")
                    # b, t -> b
                    y_trues.append(target.cpu().detach().numpy().reshape(-1, x.shape[-1]))
                    y_hats.append(x.cpu().detach().numpy().reshape(-1, x.shape[-1]))
                scores.append(score.cpu().detach().numpy().reshape(-1, score.shape[-1]))
        y_hats = np.concatenate(y_hats, axis=0)
        scores = np.concatenate(scores, axis=0).mean(axis=-1)

        label = test_dataloader.dataset.label
        scores = scores[:label.shape[0]]
        y_hats = y_hats[:label.shape[0]]

        return scores, label, y_hats

