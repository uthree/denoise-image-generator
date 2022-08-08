import torch
import torch.nn as nn
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet

class SeparatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super().__init__()
        self.c1 = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, groups=input_channels, padding_mode='replicate')
        self.c2 = nn.Conv2d(input_channels, output_channels, 1, 1, 0)

    def forward(self, x):
        return self.c2(self.c1(x))

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

        unet.downs[i][0].block1.proj = SeparatedConv2d(c, c, 7, 1, 3)
        unet.downs[i][0].block2.proj = nn.Conv2d(c, c, 1, 1, 0)
        unet.downs[i][1].block1.proj = SeparatedConv2d(c, c, 7, 1, 3)
        unet.downs[i][1].block2.proj = nn.Conv2d(c, c, 1, 1, 0)

    for i, m in enumerate(dim_mults):
        c = dim * m
        prev_c = dim * dim_mults[i-1] if i > 0 else dim * dim_mults[0]
        j = -(i+1)

        unet.ups[j][0].block1.proj = SeparatedConv2d(c+prev_c, c, 7, 1, 3)
        unet.ups[j][0].block2.proj = nn.Conv2d(c, c, 1, 1, 0)
        unet.ups[j][1].block1.proj = SeparatedConv2d(c+prev_c, c, 7, 1, 3)
        unet.ups[j][1].block2.proj = nn.Conv2d(c, c, 1, 1, 0)

    for i in range(len(dim_mults)):
        unet.downs[i][2].fn = nn.Identity()
        unet.ups[i][2].fn = nn.Identity()

    return unet
