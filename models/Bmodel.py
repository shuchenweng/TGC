import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from miscc.config import cfg


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


# Downsale the spatial size by a factor of 2
def downBlock_G(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block

def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, segmap):
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        gamma = nn.Sigmoid()(gamma)
        beta = nn.Sigmoid()(beta)
        return gamma, beta

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel, affine=False)
        self.style = nn.Linear(style_dim, in_channel * 2)
        seg_channel = cfg.GAN.P_NUM
        self.gate = SPADE(in_channel, seg_channel)

    def forward(self, input, style, seg):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)

        gam_gate, bet_gate = self.gate(seg)
        out = (gamma * gam_gate + 1) * out + beta * bet_gate

        return out

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class ResBlock_ADIN(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock_ADIN, self).__init__()
        self.block_1 = nn.Sequential(
            conv3x3(channel_num, channel_num * 2)
        )
        self.adin_1 = AdaptiveInstanceNorm(channel_num * 2, cfg.GAN.Z_DIM)
        self.block_2 = nn.Sequential(
            GLU(),
            conv3x3(channel_num, channel_num),
        )
        self.adin_2 = AdaptiveInstanceNorm(channel_num, cfg.GAN.Z_DIM)

    def forward(self, x, z_code, seg):
        residual = x
        out = self.block_1(x)
        out = self.adin_1(out, z_code, seg)
        out = self.block_2(out)
        out = self.adin_2(out, z_code, seg)
        out += residual
        return out

# Downsale the spatial size by a factor of n
def encode_image_by_ntimes(ngf, ndf, n_layer, kernel_size=4, stride=2, padding=1, up_pow=8, use_spn=False):
    sequence = []
    if use_spn:
        sequence += [spectral_norm(nn.Conv2d(3+ngf, ndf, kernel_size, stride, padding, bias=False))]
    else:
        sequence += [nn.Conv2d(3+ngf, ndf, kernel_size, stride, padding, bias=False)]
    sequence += [nn.LeakyReLU(0.2, inplace=True)]

    for n in range(1, n_layer):
        nf_mult_prev = ndf * min(2**(n-1), up_pow)
        nf_mult = ndf * min(2**n, up_pow)

        if use_spn:
            sequence += [spectral_norm(nn.Conv2d(nf_mult_prev, nf_mult, kernel_size, stride, padding, bias=False))]
        else:
            sequence += [nn.Conv2d(nf_mult_prev, nf_mult, kernel_size, stride, padding, bias=False)]
        sequence += [
            nn.BatchNorm2d(nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

    encode_img = nn.Sequential(*sequence)
    return encode_img

def cos_attention(A,B):
    batch_size = A.shape[0]
    idf = A.shape[1]
    A_dim = A.shape[2]
    B_dim = B.shape[2]
    A = A.permute((0,2,1)).reshape(batch_size*A_dim,idf)
    A = A / (A.norm(p=2, dim=1).reshape(-1, 1).expand_as(A)+1e-8)
    A = A.reshape(batch_size,A_dim,idf)
    B = B.permute((0,2,1)).reshape(batch_size*B_dim,idf)
    B = B / (B.norm(p=2, dim=1).reshape(-1, 1).expand_as(B)+1e-8)
    B = B.reshape(batch_size,B_dim,idf).permute((0,2,1))
    attn = torch.bmm(A,B) + 1
    return attn

def dot_attention(A,B):
    A = A.permute((0,2,1))
    attn = torch.bmm(A,B)
    return attn