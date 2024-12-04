import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math
from .unet import *
from scipy.special import hankel1
from .eit import UnitGaussianNormalizer


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class EuclideanPositionEncoding(nn.Module):
    def __init__(self, dmodel,
                 coords_dim=2,
                 trainable=False,
                 debug=False):
        super(EuclideanPositionEncoding, self).__init__()
        """
        channel expansion for input
        """
        self.pos_proj = nn.Conv2d(coords_dim, dmodel, kernel_size=1)
        self.trainable = trainable
        self.debug = debug

    def _get_position(self, x):
        bsz, _, h, w = x.size()  # x is bsz, channel, h, w
        grid_x, grid_y = torch.linspace(0, 1, h), torch.linspace(0, 1, w)
        mesh = torch.stack(torch.meshgrid(
            [grid_x, grid_y], indexing='ij'), dim=0)
        return mesh

    def forward(self, x):
        pos = self._get_position(x).to(x.device)
        pos = pos.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        pos = self.pos_proj(pos)

        x = x + pos
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, 
                       channel_last=False, 
                       trainable=False, 
                       pos_cache=None,
                       debug=False):
        """
        modified from https://github.com/tatp22/multidim-positional-encoding
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(math.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)
        self.channel_last = channel_last
        self.trainable = trainable
        self.debug = debug
        self.pos_cache = pos_cache

    def forward(self, x):
        """
        :param x: A 4d tensor of size (batch_size, C, h, w)
        :return: Positional Encoding Matrix of size (batch_size, C, x, y)
        """
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)

        if self.pos_cache is not None and self.pos_cache.shape == x.shape:
            return self.pos_cache + x

        bsz, n_channel, h, w = x.shape
        pos_x = torch.arange(h, device=x.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(w, device=x.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        sin_inp_y = torch.einsum("i,j->ij", self.inv_freq, pos_y)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=0).unsqueeze(-1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()),
                          dim=0).unsqueeze(-2)

        emb = torch.zeros((self.channels * 2, h, w),
                          device=x.device, dtype=x.dtype)
        emb[:self.channels, ...] = emb_x
        emb[self.channels:2 * self.channels, ...] = emb_y

        emb = emb[:n_channel, ...].unsqueeze(0).repeat(bsz, 1, 1, 1)

        if self.channel_last:
            emb = emb.permute(0, 2, 3, 1)

        self.pos_cache = emb
        return self.pos_cache + x


class Attention(nn.Module):
    def __init__(self, dim,
                 heads=4,
                 dim_head=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 bias=False,
                 norm_type='layer',
                 skip_connection=True,
                 return_attention=False,
                 softmax=True,
                 sinosoidal_pe=False,
                 pe_trainable=False,
                 debug=False):
        super(Attention, self).__init__()

        self.heads = heads
        self.dim_head = dim // heads * 2 if dim_head is None else dim_head
        self.inner_dim = self.dim_head * heads  # like dim_feedforward
        self.attn_factor = self.dim_head ** (-0.5)
        self.bias = bias
        self.softmax = softmax
        self.skip_connection = skip_connection
        self.return_attention = return_attention
        self.debug = debug

        self.to_qkv = depthwise_separable_conv(
            dim, self.inner_dim*3, bias=self.bias)
        self.to_out = depthwise_separable_conv(
            self.inner_dim, dim, bias=self.bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        PE = PositionalEncoding2D if sinosoidal_pe else EuclideanPositionEncoding
        self.pe = PE(dim, trainable=pe_trainable)
        self.norm_type = norm_type
        self.norm_q = self._get_norm(self.dim_head, self.heads,
                                     eps=1e-6)

        self.norm_k = self._get_norm(self.dim_head, self.heads,
                                     eps=1e-6)

        self.norm_out = self._get_norm(self.dim_head, self.heads,
                                       eps=1e-6)
        self.norm = LayerNorm2d(dim, eps=1e-6)

    def _get_norm(self, dim, n_head, **kwargs):
        if self.norm_type == 'layer':
            norm = nn.LayerNorm
        elif self.norm_type == "batch":
            norm = nn.BatchNorm1d
        elif self.norm_type == "instance":
            norm = nn.InstanceNorm1d
        else:
            norm = Identity
        return nn.ModuleList(
            [copy.deepcopy(norm(dim, **kwargs)) for _ in range(n_head)])

    def forward(self, x):

        _, _, h, w = x.size()
        x = self.pe(x)

        #B, inner_dim, H, W
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head',
                                          dim_head=self.dim_head,
                                          heads=self.heads,
                                          h=h, w=w), (q, k, v))

        q = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_q, (q[:, i, ...] for i in range(self.heads)))], dim=1)
        k = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_k, (k[:, i, ...] for i in range(self.heads)))], dim=1)

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        q_k_attn *= self.attn_factor
        if self.softmax:
            q_k_attn = F.softmax(q_k_attn, dim=-1)
        else:
            q_k_attn /= (h*w)

        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)

        if self.skip_connection:
            out = out + v

        out = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_out, (out[:, i, ...] for i in range(self.heads)))], dim=1)

        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w',
                        h=h, w=w, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)

        out = self.proj_drop(out)

        out = self.norm(out)

        if self.return_attention:
            return out, q_k_attn
        else:
            return out


class CrossConv(nn.Module):
    def __init__(self, dim, dim_c,
                 scale_factor=2):
        super(CrossConv, self).__init__()

        self.dim = dim  # dim = C
        self.dim_c = dim_c  # dim_c = 2*C
        self.convt = nn.ConvTranspose2d(
            dim_c, dim, scale_factor, stride=scale_factor)

    def forward(self, xf, xc):
        x = self.convt(xc)
        x = torch.cat([xf, x], dim=1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim,
                 dim_c,
                 scale_factor=[2, 2],
                 heads=4,
                 dim_head=64,
                 attn_drop=0.1,
                 proj_drop=0.1,
                 skip_connection=False,
                 hadamard=False,
                 softmax=True,
                 pe_trainable=False,
                 sinosoidal_pe=False,
                 bias=False,
                 return_attn=False,
                 debug=False):
        super(CrossAttention, self).__init__()

        self.heads = heads
        self.dim_head = dim // heads * 2 if dim_head is None else dim_head
        self.inner_dim = self.dim_head * heads  # like dim_feedforward
        self.c2f_factor = scale_factor
        self.f2c_factor = [1/s for s in scale_factor]
        self.attn_factor = self.dim_head ** (-0.5)
        self.bias = bias
        self.hadamard = hadamard
        self.softmax = softmax
        self.skip_connection = skip_connection
        self.return_attn = return_attn
        self.debug = debug
        self.dim = dim
        self.dim_c = dim_c

        self.q_proj = depthwise_separable_conv(
            self.dim_c, self.inner_dim, bias=self.bias)
        self.k_proj = depthwise_separable_conv(
            self.dim_c, self.inner_dim, bias=self.bias)
        self.v_proj = depthwise_separable_conv(
            self.dim, self.inner_dim, bias=self.bias)
        self.out_proj = depthwise_separable_conv(
            self.inner_dim, self.dim, bias=self.bias)
        self.skip_proj = depthwise_separable_conv(
            self.dim_c, self.dim, bias=self.bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        PE = PositionalEncoding2D if sinosoidal_pe else EuclideanPositionEncoding
        self.pe = PE(self.dim, trainable=pe_trainable)
        self.pe_c = PE(self.dim_c, trainable=pe_trainable)
        self.norm_k = self._get_norm(self.dim_head, self.heads, eps=1e-6)
        self.norm_q = self._get_norm(self.dim_head, self.heads, eps=1e-6)
        self.norm_out = LayerNorm2d(2*self.dim, eps=1e-6)

    def _get_norm(self, dim, n_head, norm=None, **kwargs):
        norm = nn.LayerNorm if norm is None else norm
        return nn.ModuleList(
            [copy.deepcopy(norm(dim, **kwargs)) for _ in range(n_head)])

    def forward(self, xf, xc):

        _, _, hf, wf = xf.size()
        xf = self.pe(xf)

        _, _, ha, wa = xc.size()
        xc = self.pe_c(xc)

        #B, inner_dim, H, W
        q = self.q_proj(xc)
        k = self.k_proj(xc)
        v = self.v_proj(xf)

        res = self.skip_proj(xc)
        res = F.interpolate(res, scale_factor=self.c2f_factor,
                            mode='bilinear',
                            align_corners=True,
                            recompute_scale_factor=True)

        q, k = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head',
                   dim_head=self.dim_head, heads=self.heads, h=ha, w=wa), (q, k))

        v = F.interpolate(v, scale_factor=self.f2c_factor,
                          mode='bilinear',
                          align_corners=True,
                          recompute_scale_factor=True)
        v = rearrange(v, 'b (dim_head heads) h w -> b heads (h w) dim_head',
                      dim_head=self.dim_head, heads=self.heads, h=ha, w=wa)

        q = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_q, (q[:, i, ...] for i in range(self.heads)))], dim=1)
        k = torch.stack(
            [norm(x) for norm, x in
             zip(self.norm_k, (k[:, i, ...] for i in range(self.heads)))], dim=1)

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        q_k_attn *= self.attn_factor

        if self.softmax:
            q_k_attn = F.softmax(q_k_attn, dim=-1)
        else:
            q_k_attn /= (ha*wa)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w',
                        h=ha, w=wa, dim_head=self.dim_head, heads=self.heads)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        out = F.interpolate(out, scale_factor=self.c2f_factor,
                            mode='bilinear',
                            align_corners=True,
                            recompute_scale_factor=True)

        if self.hadamard:
            out = torch.sigmoid(out)
            out = out*xf

        if self.skip_connection:
            out = out+xf

        out = torch.cat([out, res], dim=1)

        out = self.norm_out(out)

        if self.return_attn:
            return out, q_k_attn
        else:
            return out


class DownBlock(nn.Module):
    """Downscaling with interp then double conv"""

    def __init__(self, in_channels,
                 out_channels,
                 scale_factor=[0.5, 0.5],
                 batch_norm=True,
                 activation='relu'):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = DoubleConv(in_channels, out_channels,
                               batch_norm=batch_norm,
                               activation=activation)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor,
                          mode='bilinear',
                          align_corners=True,
                          recompute_scale_factor=True)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, nc_coarse, nc_fine,
                 heads=4,
                 activation='relu',
                 hadamard=False,
                 attention=True,
                 softmax=True,
                 skip_connection=False,
                 sinosoidal_pe=False,
                 pe_trainable=False,
                 batch_norm=False,
                 debug=False):
        super(UpBlock, self).__init__()
        if attention:
            self.up = CrossAttention(nc_fine,
                                     nc_coarse,
                                     heads=heads,
                                     dim_head=nc_coarse//2,
                                     skip_connection=skip_connection,
                                     hadamard=hadamard,
                                     softmax=softmax,
                                     sinosoidal_pe=sinosoidal_pe,
                                     pe_trainable=pe_trainable)
        else:
            self.up = CrossConv(nc_fine, nc_coarse)

        self.conv = DoubleConv(nc_coarse, nc_fine,
                               batch_norm=batch_norm,
                               activation=activation)
        self.debug = debug

    def forward(self, xf, xc):
        x = self.up(xf, xc)
        x = self.conv(x)
        return x


class UTransformer(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 dim=64,
                 heads=4,
                 input_size=(224, 224),
                 activation='gelu',
                 attention_coarse=True,
                 attention_up=True,
                 batch_norm=False,
                 attn_norm_type='layer',
                 skip_connection=True,
                 softmax=True,
                 pe_trainable=False,
                 hadamard=False,
                 sinosoidal_pe=False,
                 add_grad_channel=True,
                 out_latent=False,
                 debug=False,
                 **kwargs):
        super(UTransformer, self).__init__()

        self.inc = DoubleConv(in_channels, dim,
                              activation=activation,
                              batch_norm=batch_norm)
        self.down1 = DownBlock(dim, 2*dim,
                               activation=activation,
                               batch_norm=batch_norm)
        self.down2 = DownBlock(2*dim, 4*dim,
                               activation=activation,
                               batch_norm=batch_norm)
        self.down3 = DownBlock(4*dim, 8*dim,
                               activation=activation,
                               batch_norm=batch_norm)
        if attention_coarse:
            self.up0 = Attention(8*dim, heads=heads,
                                 softmax=softmax,
                                 norm_type=attn_norm_type,
                                 sinosoidal_pe=sinosoidal_pe,
                                 pe_trainable=pe_trainable,
                                 skip_connection=skip_connection)
        else:
            self.up0 = DoubleConv(8*dim, 8*dim,
                                  activation=activation,
                                  batch_norm=batch_norm)
        self.up1 = UpBlock(8*dim, 4*dim,
                           heads=heads,
                           batch_norm=batch_norm,
                           hadamard=hadamard,
                           attention=attention_up,
                           softmax=softmax,
                           sinosoidal_pe=sinosoidal_pe,
                           pe_trainable=pe_trainable)
        self.up2 = UpBlock(4*dim, 2*dim,
                           heads=heads,
                           batch_norm=batch_norm,
                           hadamard=hadamard,
                           attention=attention_up,
                           softmax=softmax,
                           sinosoidal_pe=sinosoidal_pe,
                           pe_trainable=pe_trainable)
        self.up3 = UpBlock(2*dim, dim,
                           heads=heads,
                           batch_norm=batch_norm,
                           hadamard=hadamard,
                           attention=attention_up,
                           softmax=softmax,
                           sinosoidal_pe=sinosoidal_pe,
                           pe_trainable=pe_trainable,)
        self.out = OutConv(dim, out_channels)
        self.out_latent = out_latent
        self.add_grad_channel = add_grad_channel
        self.input_size = input_size
        self.debug = debug

    # def forward(self, x, gradx, *args, **kwargs):
    #     "input dim: bsz, n, n, C"
    #     if gradx.ndim == x.ndim and self.add_grad_channel:
    #         x = torch.cat([x, gradx], dim=-1)
    #     x = x.permute(0, 3, 1, 2)
        
    def forward(self, x_gradx, *args, **kwargs):
        
        x_gradx = x_gradx.to(torch.float32)
        x = x_gradx.permute(0, 3, 1, 2)

        if self.input_size:
            _, _, *size = x.size()
            x = F.interpolate(x, size=self.input_size,
                              mode='bilinear',
                              align_corners=True)

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x4 = self.up0(x4)

        x = self.up1(x3, x4)

        x = self.up2(x2, x)

        x = self.up3(x1, x)

        out = self.out(x)

        if self.input_size:
            out = F.interpolate(out, size=size,
                                mode='bilinear',
                                align_corners=True)

            out = out.permute(0, 2, 3, 1)

        if self.out_latent:
            return dict(preds=out,
                        preds_latent=[x2, x3, x4])
        else:
            return dict(preds=out)
        
class MiniUTransformer(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 dim=64,
                 heads=4,
                 input_size=(224, 224),
                 activation='gelu',
                 attention_coarse=True,
                 attention_up=True,
                 batch_norm=False,
                 attn_norm_type='layer',
                 skip_connection=True,
                 softmax=True,
                 pe_trainable=False,
                 hadamard=False,
                 sinosoidal_pe=False,
                 add_grad_channel=True,
                 out_latent=False,
                 debug=False,
                 **kwargs):
        super(MiniUTransformer, self).__init__()

        self.inc = DoubleConv(in_channels, dim,
                              activation=activation,
                              batch_norm=batch_norm)
        self.down1 = DownBlock(dim, 2*dim,
                               activation=activation,
                               batch_norm=batch_norm)
        self.down2 = DownBlock(2*dim, 4*dim,
                               activation=activation,
                               batch_norm=batch_norm)
        # self.down3 = DownBlock(4*dim, 8*dim,
        #                        activation=activation,
        #                        batch_norm=batch_norm)
        if attention_coarse:
            self.up0 = Attention(4*dim, heads=heads,
                                 softmax=softmax,
                                 norm_type=attn_norm_type,
                                 sinosoidal_pe=sinosoidal_pe,
                                 pe_trainable=pe_trainable,
                                 skip_connection=skip_connection)
        else:
            self.up0 = DoubleConv(4*dim, 4*dim,
                                  activation=activation,
                                  batch_norm=batch_norm)
        # self.up1 = UpBlock(8*dim, 4*dim,
        #                    heads=heads,
        #                    batch_norm=batch_norm,
        #                    hadamard=hadamard,
        #                    attention=attention_up,
        #                    softmax=softmax,
        #                    sinosoidal_pe=sinosoidal_pe,
        #                    pe_trainable=pe_trainable)
        self.up1= UpBlock(4*dim, 2*dim,
                           heads=heads,
                           batch_norm=batch_norm,
                           hadamard=hadamard,
                           attention=attention_up,
                           softmax=softmax,
                           sinosoidal_pe=sinosoidal_pe,
                           pe_trainable=pe_trainable)
        self.up2 = UpBlock(2*dim, dim,
                           heads=heads,
                           batch_norm=batch_norm,
                           hadamard=hadamard,
                           attention=attention_up,
                           softmax=softmax,
                           sinosoidal_pe=sinosoidal_pe,
                           pe_trainable=pe_trainable,)
        self.out = OutConv(dim, out_channels)
        self.out_latent = out_latent
        self.add_grad_channel = add_grad_channel
        self.input_size = input_size
        self.debug = debug

    # def forward(self, x, gradx, *args, **kwargs):
    #     "input dim: bsz, n, n, C"
    #     if gradx.ndim == x.ndim and self.add_grad_channel:
    #         x = torch.cat([x, gradx], dim=-1)
    #     x = x.permute(0, 3, 1, 2)
        
    def forward(self, x_gradx, *args, **kwargs):
 
        x_gradx = x_gradx.to(torch.float32)
        x = x_gradx.permute(0, 3, 1, 2)

        if self.input_size:
            _, _, *size = x.size()
            x = F.interpolate(x, size=self.input_size,
                              mode='bilinear',
                              align_corners=True)

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x3 = self.up0(x3)

        x = self.up1(x2, x3)

        x = self.up2(x1, x)

        out = self.out(x)

        if self.input_size:
            out = F.interpolate(out, size=size,
                                mode='bilinear',
                                align_corners=True)

            out = out.permute(0, 2, 3, 1)

        if self.out_latent:
            return dict(preds=out,
                        preds_latent=[x2, x3])
        else:
            return dict(preds=out)
        

class DSMReconstructionGenerator(nn.Module):
    def __init__(self, 
        s_initial_value,
        train_s:bool, 
        g_f_real,
        g_f_imag,
        v,
        w,
        v_inv,
        device
        ):
        super(DSMReconstructionGenerator, self).__init__()
        self.num_of_channels = g_f_real.shape[-1]
        self.train_s = train_s
        #多通道s
        if  isinstance(s_initial_value, list):
            self.multiple_s = True
            s_list = nn.ParameterList()
            for item in torch.tensor(s_initial_value):
                s_list.append(item)
            self.s = s_list
        #单通道s
        elif isinstance(s_initial_value, (int, float)):
            self.multiple_s = False
            self.s = nn.Parameter(torch.Tensor([s_initial_value]), requires_grad=self.train_s)
        else:
            raise ValueError("Input must be either a list or a number.")
        self.g_f_real = g_f_real.to(device).to(torch.float32)   #[r, n*n, C]
        self.g_f_imag = g_f_imag.to(device).to(torch.float32)
        self.v = v.to(device).to(torch.float32)
        self.w = w.to(device).to(torch.float32)
        self.v_inv = v_inv.to(device).to(torch.float32)
        self.device = device

    def forward(self, u_s,  *args, **kwargs):
        "u_s dim: (bsz, r, C)"
        C = self.num_of_channels
        u_s_real = (u_s[0][:, ::10, :]).to(self.device).to(torch.float32)
        u_s_imag = (u_s[1][:, ::10, :]).to(self.device).to(torch.float32)
        # u_s_real = (u_s[0]).to(self.device).to(torch.float32)
        # u_s_imag = (u_s[1]).to(self.device).to(torch.float32)
        g_f = torch.complex(self.g_f_real, self.g_f_imag)
        if self.multiple_s:
            lb_real = torch.zeros_like(u_s_real, dtype=torch.float32)
            lb_imag = torch.zeros_like(u_s_imag, dtype=torch.float32)
            for i in range(len(self.s)):
                Lam = torch.diag(torch.pow(self.w, self.s[i]))
                # lb_real[..., i:i+1] = self.v@Lam@self.v_inv@u_s_real[..., i:i+1]
                # lb_imag[..., i:i+1] = self.v@Lam@self.v_inv@u_s_imag[..., i:i+1]
                lb_real[..., i:i+1] = torch.einsum('ij, ajk->aik', self.v@Lam@self.v_inv, u_s_real[..., i:i+1])
                lb_imag[..., i:i+1] = torch.einsum('ij, ajk->aik', self.v@Lam@self.v_inv, u_s_imag[..., i:i+1])
        else:
            Lam = torch.diag(torch.pow(self.w, self.s))
            # lb_real = torch.einsum('ij, aik->ajk', self.v@Lam@self.v_inv, u_s_real)
            # lb_imag = torch.einsum('ij, aik->ajk', self.v@Lam@self.v_inv, u_s_imag)
            lb_real = torch.einsum('ij, ajk->aik', self.v@Lam@self.v_inv, u_s_real)
            lb_imag = torch.einsum('ij, ajk->aik', self.v@Lam@self.v_inv, u_s_imag)
            # lb_imag = u_s_imag
        
        lb_u_s = torch.complex(lb_real, lb_imag)
        u_s = torch.complex(u_s_real, u_s_imag)
        bsz = u_s_real.shape[0]
        n = int(math.sqrt(g_f.shape[-2]))
        
        phi_real = torch.einsum('bic, ijc -> bjc', lb_real, torch.real(g_f.conj()))\
              - torch.einsum('bic, ijc -> bjc', lb_imag, torch.imag(g_f.conj()))
        phi_imag = torch.einsum('bic, ijc -> bjc', lb_real, torch.imag(g_f.conj()))\
              + torch.einsum('bic, ijc -> bjc', lb_imag, torch.real(g_f.conj()))
        phi = torch.complex(phi_real, phi_imag)
        phi_ = torch.sqrt(phi_real**2 + phi_imag**2)
        u_s_norm = torch.norm(lb_u_s, dim=1)
        g_f_norm = torch.norm(g_f, dim=0)
        # print(g_f_norm.shape)
        norm = torch.einsum('bc, kc -> bkc', u_s_norm, g_f_norm)
        # out = (phi_ / norm).reshape(bsz, n, n, C)  
        out = phi_.reshape(bsz, n, n, C)
        return out 
    
        # phi_real = torch.einsum('bic, ijc -> bjc', lb_real, torch.real(g_f.conj()))\
        #         - torch.einsum('bic, ijc -> bjc', lb_imag, torch.imag(g_f.conj()))
        #     phi_imag = torch.einsum('bic, ijc -> bjc', lb_real, torch.imag(g_f.conj()))\
        #         + torch.einsum('bic, ijc -> bjc', lb_imag, torch.real(g_f.conj()))
        #     phi = torch.norm(torch.complex(phi_real, phi_imag), dim=1).unsqueeze(1)
        #     u_s_norm = torch.norm(u_s, dim=1).unsqueeze(1)
        #     g_f_norm = torch.norm(g_f, dim=0)
        #     norm = torch.einsum('bij, kj -> bkj', u_s_norm, g_f_norm)
        #     out = (phi / norm).reshape(bsz, n, n, C)  
        #     return out
    


class DataPreprocessor(nn.Module):
    def __init__(self,
                 normalizer_x=None,
                 normalization=False,
                 subsample: int = 1,
                 subsample_attn: int = 4,
                 subsample_method = 'nearest',
                 channel = 3,
                 train_data = True,
                 online_grad=True,
                 return_grad=True,
                 return_boundary=True
                  ):

        super(DataPreprocessor, self).__init__()

        self.normalizer_x = normalizer_x
        self.normalization = normalization
        self.subsample = subsample
        self.subsample_attn = subsample_attn
        self.subsample_method = subsample_method
        self.channel = channel
        self.train_data = train_data
        self.online_grad = online_grad
        self.return_grad = return_grad
        self.return_boundary = return_boundary
        
        self.n_grid_fine = 64
        self.n_grid = int(((self.n_grid_fine - 1)/self.subsample) + 1)
        self.n_grid_coarse = int(
            ((self.n_grid_fine - 1)/self.subsample_attn) + 1)
        self.h = 1/self.n_grid_fine

        self.grid_c = self.get_grid(self.n_grid_coarse)
        self.grid = self.get_grid(self.n_grid_fine,
                                  subsample=self.subsample,
                                  return_boundary=self.return_boundary)
        if self.train_data and self.normalization:
            self.normalizer_x = UnitGaussianNormalizer()
            self.normalizer_y = UnitGaussianNormalizer()
            self.phi = self.normalizer_x.fit_transform(self.phi)
    
    @staticmethod
    def get_grad(f, h):
        '''
        h: mesh size
        n: grid size
        separate input for online grad generating
        input f: (N, C, n, n)
        '''
        bsz, N_C = f.shape[:2]
        # n = int(((n - 1)/s) + 1)
        # f = torch.tensor(f, dtype=torch.complex128)
        fx, fy = [], []
        for i in range(N_C):
            '''smaller mem footprint'''
            _fx, _fy = DataPreprocessor.central_diff(f[:, i], h)
            fx.append(_fx)
            fy.append(_fy)
        gradf = torch.stack(fx+fy, dim=1)  # (N, 2*C, n, n)
        return gradf
    
    @staticmethod
    def central_diff(f, h, mode='constant', padding=True):
        """
        mode: see
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        # x: (batch, n, n)
        # b = x.shape[0]
        # f_tensor = torch.tensor(f, dtype=torch.complex128)
    # Add padding if required
        if padding:
            f = torch.nn.functional.pad(f, (1, 1, 1, 1), mode=mode, value=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (f[:, d:, s:-s] - f[:, :-d, s:-s]) / d
        grad_y = (f[:, s:-s, d:] - f[:, s:-s, :-d]) / d

        return grad_x/h, grad_y/h

    @staticmethod
    def get_grid(n_grid, subsample=1, return_boundary=True):
        x = torch.linspace(0, 1, n_grid)
        y = torch.linspace(0, 1, n_grid)
        x, y = torch.meshgrid(x, y, indexing='ij')
        s = subsample

        if return_boundary:
            x = x[::s, ::s]
            y = y[::s, ::s]
        else:
            x = x[::s, ::s][1:-1, 1:-1]
            y = y[::s, ::s][1:-1, 1:-1]

        grid = torch.stack([x, y], dim=-1)
    
        # grid is DoF, excluding boundary (n, n, 2), or (n-2, n-2, 2)
        return grid
    
    def forward(self, dsm_phi, *args, **kwargs):
        bsz = dsm_phi.shape[0]
        s = self.subsample

        if self.return_grad and not self.online_grad:
            gradphi = self.get_grad(dsm_phi, self.h)  # u is (bsz, C, n, n)
            gradphi = gradphi[..., ::s, ::s]
            gradphi = gradphi.transpose((0, 2, 3, 1)) # (N, n, n, C)
        else:
            gradphi = np.zeros((bsz, ))  # placeholder
        self.gradphi = gradphi
        
        if self.train_data and self.normalization:
            self.normalizer_x = UnitGaussianNormalizer()
            self.normalizer_y = UnitGaussianNormalizer()
            dsm_phi = self.normalizer_x.fit_transform(dsm_phi)

            # if self.return_boundary:
            #     _ = self.normalizer_y.fit_transform(x=targets)
            # else:
            #     _ = self.normalizer_y.fit_transform(
            #         x=targets[:, 1:-1, 1:-1, :])
        elif self.normalization:
            dsm_phi = self.normalizer_x.transform(dsm_phi)

        pos_dim = 2
        # uniform grid fine for all samples (n, n, 2)
        if self.subsample_attn is None:
            grid_c = torch.tensor([1.0])  # place holder
        else:
            grid_c = self.grid_c.reshape(-1, pos_dim)  # (n_s*n_s, 2)
        
        grid = self.grid
        # dsm_phi = dsm_phi[..., ::s, ::s]
        # dsm_phi = dsm_phi.permute((0, 2, 3, 1)) # (N, n, n, C)
        
        if self.return_grad and self.online_grad:
            dsm_phi = dsm_phi.permute(0, 3, 1, 2)
            # dsm_phi (bsz, C, n, n)
            gradphi = self.get_grad(dsm_phi, self.h)#(bsz, 2*C, n, n)
            gradphi = gradphi.permute(0, 2, 3, 1) #(bsz, n, n, 2*C)
            dsm_phi = dsm_phi[..., ::s, ::s]
            dsm_phi = dsm_phi.permute((0, 2, 3, 1))
            return torch.cat([dsm_phi, gradphi], dim=-1)
        elif self.return_grad:
            dsm_phi = dsm_phi[..., ::s, ::s]
            gradphi = self.gradphi
            return torch.cat([dsm_phi, gradphi], dim=-1)
        else:
            dsm_phi = dsm_phi[..., ::s, ::s]
            gradphi = torch.tensor(float('nan'))
            return dsm_phi
