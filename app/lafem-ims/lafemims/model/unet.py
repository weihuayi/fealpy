""" Parts of the U-Net model and basic block for ResNet
https://github.com/milesial/Pytorch-UNet
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
https://github.com/Dootmaan/MT-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops import rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, zeros_

class Identity(nn.Module):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
    not used anymore as
    https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)
        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        return self.id(x)

class LayerNorm2d(nn.Module):
    """
    a wrapper for GroupNorm
    https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    input and output shapes: (bsz, C, H, W)
    """
    def __init__(self, num_channels, 
                       eps=1e-06, 
                       elementwise_affine=True, 
                       device=None, 
                       dtype=None) -> None:
        super(LayerNorm2d, self).__init__()
        self.norm =  nn.GroupNorm(1, num_channels, 
                                  eps=eps,
                                  affine=elementwise_affine, 
                                  device=device, 
                                  dtype=dtype)
    def forward(self, x):
        return self.norm(x)


class DoubleConv(nn.Module):
    """(convolution => BN or LN=> ReLU) * 2"""

    def __init__(self, in_channels,
                 out_channels,
                 mid_channels=None,
                 activation='relu',
                 batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if batch_norm:
            self.norm1 = nn.BatchNorm2d(mid_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = LayerNorm2d(mid_channels)
            self.norm2 = LayerNorm2d(out_channels)

        if activation == 'silu':
            self.activation1 = nn.SiLU()
            self.activation2 = nn.SiLU()
        elif activation == 'gelu':
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        else:
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation2(x)


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False, xavier_init=1e-2):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.xavier_init = xavier_init
        self._reset_parameters()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

    def _reset_parameters(self):
        for layer in [self.depthwise, self.pointwise]:
            for param in layer.parameters():
                if param.ndim > 1:
                    xavier_uniform_(param, gain=self.xavier_init)
                else:
                    constant_(param, 0)

class UNet(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 dim=64,
                 encoder_layers=5,
                 decoder_layers=None,
                 scale_factor=2,
                 input_size=(224,224),
                 return_latent=False,
                 debug=False,
                 **kwargs):
        '''new implementation for eit-transformer paper'''
        super(UNet, self).__init__()
        self.layers = decoder_layers if decoder_layers else encoder_layers
        enc_dim = [dim*2**(k-1) for k in range(1, encoder_layers+1)]
        if decoder_layers:
            dec_dim = [dim*2**(k-1) for k in range(decoder_layers, 0, -1)]
        else:
            dec_dim = enc_dim[::-1]
        enc_dim = [in_channels] + enc_dim
        self.encoder = nn.ModuleList(
            [DoubleConv(enc_dim[i], enc_dim[i+1]) for i in range(encoder_layers)])
        self.pool = nn.MaxPool2d(scale_factor)

        self.up_blocks  = nn.ModuleList([nn.ConvTranspose2d(dec_dim[i], dec_dim[i+1], scale_factor, scale_factor) for i in range(encoder_layers-1)])
        self.decoder = nn.ModuleList([DoubleConv(dec_dim[i], dec_dim[i+1]) for i in range(len(dec_dim)-1)])
        self.outconv = nn.Conv2d(dec_dim[-1], out_channels, 1)

        self.input_size = input_size
        self.return_latent = return_latent
        self.debug = debug

    # def forward(self, x, xp=None, pos=None, grid=None):  # 101x101x20
    def forward(self, x, pos=None, grid=None):

        # if xp.ndim == x.ndim:
        #     x = torch.cat([x, xp], dim=-1)
        x = x.permute(0, 3, 1, 2)

        if self.input_size:
            _, _, *size = x.size()
            x = F.interpolate(x, size=self.input_size,
                              mode='bilinear',
                              align_corners=True)

        latent_enc = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            latent_enc.append(x)
            if i < self.layers-1:
                x = self.pool(x)

        latent_enc = latent_enc[::-1][1:]

        if self.debug:
            for i, z in enumerate(latent_enc):
                print(f"{i+1}-th latent from encoder: \t {z.size()}")

        for i, (up, dec) in enumerate(zip(self.up_blocks, self.decoder)):
            x = up(x)
            x  = torch.cat([x, latent_enc[i]], dim=1)
            x  = dec(x)
        
        x = self.outconv(x)

        if self.input_size:
            x = F.interpolate(x, size=size,
                                mode='bilinear',
                                align_corners=True)

        if self.return_latent:
            return dict(preds=x.permute(0, 2, 3, 1),
                    preds_latent=[z.permute(0, 2, 3, 1) for z in latent_enc])
        else:
            return dict(preds=x.permute(0, 2, 3, 1))
