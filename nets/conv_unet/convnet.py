import torch
import math

from typing import Sequence
from torch import nn, Tensor
from torch.nn import functional as F


class ConvNet(nn.Module):
    ''' A new ConvNet backbone '''

    def __init__(self,
                 in_channels: int,
                 depths: Sequence[int] = (3, 3, 3, 3),
                 feature_size: int = 72,
                 spatial_dim: int = 3,
                 drop_path_rate: float = 0.0):
        ''' Args:
        * `in_channels`: dimension of input channels.
        * `depths`: number of blocks in each stage.
        * `feature_size`: output channels of the PatchEmbed layer.
        * `spatial_dim`: number of spatial dimensions.
        * `drop_path_rate`: stochastic depth rate.
        '''
        super(ConvNet, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")
        
        self.down_samplers = nn.ModuleList()
        self.down_samplers.append(nn.Sequential(    # patch embed
            Conv(in_channels, feature_size, 4, stride=2, padding=1),
            LayerNorm(feature_size, eps=1e-6, channels_last=False)
        )) 
        for i in range(3):      # 3 intermediate downsampling conv
            dim = (2 ** i) * feature_size
            self.down_samplers.append(nn.Sequential(
                LayerNorm(dim, eps=1e-6, channels_last=False),
                Conv(dim, 2 * dim, 4, stride=2, padding=1)
            ))

        self.stages = nn.ModuleList()
        dp_rates = [r.item() for r in 
                    torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0         # counting the number of blocks
        for i in range(4):      # 4 feature resolution stages
            dim = (2 ** i) * feature_size
            self.stages.append(nn.Sequential(
                *[ConvBlock(in_dim=dim,
                            spatial_dim=spatial_dim,
                            drop_path_rate=dp_rates[cur + j])
                  for j in range(depths[i])]
            ))
            cur += depths[i]

    def forward(self, x: Tensor):
        outputs = list()        # 记录所有stage的输出
        for i in range(4):
            x = self.down_samplers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs


class ConvBlock(nn.Module):
    ''' Basic ConvNet Block '''

    def __init__(self,
                 in_dim: int,
                 drop_path_rate: float = 0.0,
                 spatial_dim: int = 3):
        ''' Args:
        * `in_dim`: dimension of input channels.
        * `drop_path_rate`: stochastic depth rate.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(ConvBlock, self).__init__()
        if spatial_dim in [2, 3]:
            Conv = nn.Conv3d if spatial_dim == 3 else nn.Conv2d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.norm = LayerNorm(in_dim, eps=1e-6, channels_last=False)
        self.attn = nn.Sequential(
            Conv(in_dim, in_dim, 1, groups=in_dim),
            nn.GELU(),
            Conv(in_dim, in_dim, 4, padding=3, groups=in_dim, dilation=2),
        )
        self.val = Conv(in_dim, in_dim, 1, groups=in_dim)
        self.grn = GlobalRespNorm(in_dim, spatial_dim=spatial_dim, 
                                  eps=1e-6, channels_last=False)
        self.proj = Conv(in_dim, in_dim, 1, groups=in_dim)
        
        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x: Tensor):
        shortcut = x        # 暂存处理之前的x
        x = self.norm(x)

        qk = self.attn(x)
        v = self.val(x)
        x = self.grn(qk * v)
        x = self.proj(x)

        y = shortcut + self.drop_path(x)
        return y


class DropPath(nn.Module):
    ''' Stochastic drop paths per sample for residual blocks. '''

    def __init__(self,
                 drop_prob: float = 0.0,
                 scale_by_keep: bool = True):
        ''' Args:
            * `drop_prob`: drop paths probability.
            * `scale_by_keep`: whether scaling by non-dropped probaility.
        '''
        super(DropPath, self).__init__()
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError("drop_path_prob should be between 0 and 1.")
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor):
        if self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0.0:
            rand_tensor.div_(keep_prob)
        return x * rand_tensor


class LayerNorm(nn.Module):
    ''' Layer Normalization '''

    def __init__(self,
                 norm_shape: int,
                 eps: float = 1e-6,
                 channels_last: bool = True):
        ''' Args:
        * `norm_shape`: dimension of the input feature.
        * `eps`: epsilon of layer normalization.
        * `channels_last`: whether the channel is the last dim.
        '''
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(norm_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(norm_shape), requires_grad=True)
        self.channels_last = channels_last
        self.norm_shape = (norm_shape,)
        self.eps = eps

    def forward(self, x: Tensor):
        if self.channels_last:  # [B, ..., C]
            y = F.layer_norm(x, self.norm_shape,
                             self.weight, self.bias, self.eps)
        else:                   # [B, C, ...]
            y = layer_norm(x, self.norm_shape, self.eps, False)
            if x.ndim == 4:
                y = self.weight[:, None, None] * y
                y += self.bias[:, None, None]
            else:
                y = self.weight[:, None, None, None] * y
                y += self.bias[:, None, None, None]
        return y


class GlobalRespNorm(nn.Module):
    ''' Global Response Normalization '''

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 channels_last: bool = False,
                 spatial_dim: int = 3):
        ''' Args:
        * `dim`: dimension of input channels.
        * `eps`: epsilon of the normalization.
        * `channels_last`: whether the channel is the last dim.
        * `spatial_dim`: number of spatial dimensions.
        '''
        super(GlobalRespNorm, self).__init__()
        if spatial_dim == 2:
            if channels_last:
                size, self.dims = [1, 1, 1, dim], (1, 2)
            else:
                size, self.dims = [1, dim, 1, 1], (2, 3)
        else:
            if channels_last:
                size, self.dims = [1, 1, 1, 1, dim], (1, 2, 3)
            else:
                size, self.dims = [1, dim, 1, 1, 1], (2, 3, 4)

        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(*size))
        self.beta = nn.Parameter(torch.zeros(*size))

    def forward(self, x: Tensor):
        ''' Args:
        * `x`: a input tensor in [B, C, (D,) W, H] shape.
        '''
        gx = torch.norm(x, p=2, dim=self.dims, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
        return x


def trunc_normal_(tsr: Tensor,
                  mean: float = 0.0,
                  std: float = 1.0,
                  a: float = -2.0,
                  b: float = 2.0):
    """ Tensor initialization with truncated normal distribution.
        * `tsr`: an n-dimensional `Tensor`.
        * `mean`: the mean of the normal distribution.
        * `std`: the standard deviation of the normal distribution.
        * `a`: the minimum cutoff value.
        * `b`: the maximum cutoff value.
    """
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)
        tsr.uniform_(2 * low - 1, 2 * high - 1)
        tsr.erfinv_()
        tsr.mul_(std * math.sqrt(2.0))
        tsr.add_(mean)
        tsr.clamp_(min=a, max=b)
        return tsr


def layer_norm(x: Tensor,
               norm_shape: Sequence[int],
               eps: float = 1e-6,
               channels_last: bool = True):
    if channels_last:   # [B, ..., C]
        y = F.layer_norm(x, norm_shape, eps=eps)
    else:               # [B, C, ...]
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
    return y
