from typing import Sequence
from torch import nn, Tensor
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from .convnet import ConvNet, layer_norm


class ConvUNET(nn.Module):
    """ConvUNET (BIBM 2023)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (3, 3, 3, 3),
        feature_size: int = 36,
        spatial_dim: int = 3,
        drop_path_rate: float = 0.0,
        end_norm: bool = True,
        deep_sup: bool = False,
    ):
        """Args:
        * `in_channels`: dimension of input channels.
        * `out_channels`: dimension of output channels.
        * `depths`: number of blocks in each stage.
        * `feature_size`: output channels of the steam layer.
        * `spatial_dim`: number of spatial dimensions.
        * `drop_path_rate`: stochastic depth rate.
        * `end_norm`: whether normalizing output features in each stage.
        * `deep_sup`: whether using deep supervision to compute the loss.
        """
        super(ConvUNET, self).__init__()
        self.end_norm = end_norm
        self.deep_sup = deep_sup

        if spatial_dim == 2:
            us_mode, Conv = "bilinear", nn.Conv2d
        elif spatial_dim == 3:
            us_mode, Conv = "trilinear", nn.Conv3d
        else:
            raise ValueError("`spatial_dim` should be 2 or 3.")

        self.backbone = ConvNet(
            in_channels=in_channels,
            depths=depths,
            feature_size=feature_size,
            spatial_dim=spatial_dim,
            drop_path_rate=drop_path_rate,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size * 2,
            out_channels=feature_size * 4,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size * 4,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size * 8,
            out_channels=feature_size * 16,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        # deep supervision
        self.sup2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode=us_mode),
            Conv(feature_size*4, out_channels, 1),
        )
        self.sup1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=us_mode),
            Conv(feature_size*2, out_channels, 1),
        )
            
        self.output = UnetOutBlock(
            spatial_dims=spatial_dim,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def project(self, x: Tensor):
        if self.end_norm:
            x = layer_norm(x, (x.shape[1],), channels_last=False)
        return x.contiguous()

    def forward(self, inputs: Tensor):
        hidden_states = self.backbone(inputs)

        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.project(hidden_states[0]))
        enc3 = self.encoder3(self.project(hidden_states[1]))
        enc4 = self.encoder4(self.project(hidden_states[2]))
        dec4 = self.encoder5(self.project(hidden_states[3]))

        dec3 = self.decoder4(dec4, enc4)
        dec2 = self.decoder3(dec3, enc3)
        dec1 = self.decoder2(dec2, enc2)
        outs = self.decoder1(dec1, enc1)

        logits = self.output(outs)
        if self.deep_sup:
            return [logits, self.sup1(dec1), self.sup2(dec2)]
        else:
            return logits


if __name__ == "__main__":
    import torch

    model = ConvUNET(1, 2, deep_sup=True)
    input = torch.randn(size=(1, 1, 64, 64, 64))
    output = model(input)
    n_params = sum([param.nelement() for param in model.parameters()])
    print(model)
    print("Dimension of outputs:", output[0].shape, output[1].shape, output[2].shape)
    print("Number of parameters: %.2fM" % (n_params * 1e-6))  # 32.62M
