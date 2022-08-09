import torch
import torch.nn as nn
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(ChannelNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x): # x: [N, C, H, W]
        m = x.mean(dim=1, keepdim=True)
        s = ((x - m) ** 2).mean(dim=1, keepdim=True)
        x = (x - m) * torch.rsqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dim_ffn=None):
        super(ConvNeXtBlock, self).__init__()
        if dim_ffn == None:
            dim_ffn = input_channels * 4
        self.c1 = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, padding_mode='replicate', groups=input_channels)
        self.norm = ChannelNorm(input_channels)
        self.c2 = nn.Conv2d(input_channels, dim_ffn, 1, 1, 0)
        self.gelu = nn.GELU()
        self.c3 = nn.Conv2d(dim_ffn, output_channels, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = self.gelu(x)
        x = self.c3(x)
        return x

def init_unet(dim = 4, dim_mults=[1, 2, 4, 8, 16, 32]):
    unet = Unet(
            dim = dim,
            dim_mults = tuple(dim_mults),
            resnet_block_groups = dim,
            )
    internal_dim_mults = [dim_mults[0]] + dim_mults[:-1]
    for i, m in enumerate(internal_dim_mults):
        c = dim * m
        prev_c = dim * internal_dim_mults[i-1] if i > 0 else dim * internal_dim_mults[0]
        j = -(i+1)

        unet.downs[i][0].block1.proj = ConvNeXtBlock(c, c, 7, 1, 3)
        unet.downs[i][0].block1.act = nn.Identity()
        unet.downs[i][0].block1.norm = nn.Identity()
        unet.downs[i][0].block2.proj = ConvNeXtBlock(c, c, 7, 1, 3)
        unet.downs[i][0].block2.act = nn.Identity()
        unet.downs[i][0].block2.norm = nn.Identity()
        unet.downs[i][1].block1.proj = ConvNeXtBlock(c, c, 7, 1, 3)
        unet.downs[i][1].block1.act = nn.Identity()
        unet.downs[i][1].block1.norm = nn.Identity()
        unet.downs[i][1].block2.proj = ConvNeXtBlock(c, c, 7, 1, 3)
        unet.downs[i][1].block2.act = nn.Identity()
        unet.downs[i][1].block2.norm = nn.Identity()

    for i, m in enumerate(dim_mults):
        c = dim * m
        prev_c = dim * dim_mults[i-1] if i > 0 else dim * dim_mults[0]
        j = -(i+1)

        unet.ups[j][0].block1.proj = ConvNeXtBlock(c+prev_c, c, 7, 1, 3)
        unet.ups[i][0].block1.act = nn.Identity()
        unet.ups[i][0].block1.norm = nn.Identity()
        unet.ups[j][0].block2.proj = ConvNeXtBlock(c, c, 7, 1, 3)
        unet.ups[i][0].block2.act = nn.Identity()
        unet.ups[i][0].block2.norm = nn.Identity()
        unet.ups[j][1].block1.proj = ConvNeXtBlock(c+prev_c, c, 7, 1, 3)
        unet.ups[i][1].block1.act = nn.Identity()
        unet.ups[i][1].block1.norm = nn.Identity()
        unet.ups[j][1].block2.proj = ConvNeXtBlock(c, c, 7, 1, 3)
        unet.ups[i][1].block2.act = nn.Identity()
        unet.ups[i][1].block2.norm = nn.Identity()


    for i in range(len(dim_mults)):
        unet.downs[i][2].fn = nn.Identity()
        unet.ups[i][2].fn = nn.Identity()

    return unet
