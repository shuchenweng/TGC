import torch
import torch.nn as nn
import math
from miscc.config import cfg
from models.Bmodel import encode_image_by_ntimes, Block3x3_leakRelu, spectral_norm, downBlock_G

# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, h_code.size(2), h_code.size(3))
            # state size (ngf+egf) x 8 x 8
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output

# For 64 x 64 images
class PAT_D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code = encode_image_by_ntimes(0, ndf, 4)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code(x_var)  # 4 x 4 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code4


# For 128 x 128 images
class PAT_D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code = encode_image_by_ntimes(0, ndf, 4)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code(x_var)  # 8 x 8 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code8


# For 256 x 256 images
class PAT_D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(PAT_D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code = encode_image_by_ntimes(0, ndf, 4)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code(x_var)  # 16 x 16 x 8df (cfg.GAN.LAYER_D_NUM=4)
        return x_code16

# For 64 x 64 images
class SHP_D_NET64(nn.Module):
    def __init__(self, x_channel):
        super(SHP_D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = x_channel
        ncf = cfg.GAN.P_NUM
        self.img_code = encode_image_by_ntimes(ngf, ndf, 4)
        self.shp_code = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
                            nn.InstanceNorm2d(ngf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        # 64//cfg.GAN.LAYER_D_NUM x 64//cfg.GAN.LAYER_D_NUM x 2^(cfg.GAN.LAYER_D_NUM-1)
        x_code4 = self.img_code(x_s_var)
        return x_code4


# For 128 x 128 images
class SHP_D_NET128(nn.Module):
    def __init__(self, x_channel):
        super(SHP_D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = x_channel
        ncf = cfg.GAN.P_NUM
        self.img_code = encode_image_by_ntimes(ngf, ndf, 4)
        self.shp_code = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
                            nn.InstanceNorm2d(ngf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        # 128//cfg.GAN.LAYER_D_NUM x 128//cfg.GAN.LAYER_D_NUM x 2^(cfg.GAN.LAYER_D_NUM-1)
        x_code8 = self.img_code(x_s_var)
        return x_code8


# For 256 x 256 images
class SHP_D_NET256(nn.Module):
    def __init__(self, x_channel):
        super(SHP_D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ngf = x_channel
        ncf = cfg.GAN.P_NUM
        self.img_code = encode_image_by_ntimes(ngf, ndf, 4)
        self.shp_code = nn.Sequential(
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
                            nn.InstanceNorm2d(ngf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, x_var, s_var):
        new_s_var = self.shp_code(s_var)
        x_s_var = torch.cat([x_var, new_s_var], dim=1)
        # 256//cfg.GAN.LAYER_D_NUM x 256//cfg.GAN.LAYER_D_NUM x 2^(cfg.GAN.LAYER_D_NUM-1)
        x_code16 = self.img_code(x_s_var)
        return x_code16

class SEG_D_NET(nn.Module):
    def __init__(self):
        super(SEG_D_NET, self).__init__()
        ndf = cfg.GAN.DF_DIM
        if cfg.MODEL.COMP_NAME == 'pixel':
            self.img_code = encode_image_by_ntimes(ngf=0, ndf=ndf, n_layer=2,
                kernel_size=4, stride=2, padding=1, use_spn=True)
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * 2, 1, 3, padding=1)),
                nn.Sigmoid()
            )

        elif cfg.MODEL.COMP_NAME == 'glpuzzle':
            for num in cfg.GAN.PUZZLE_NUM:
                layer_num = int(8 - math.log2(num))
                self.__dict__['_modules'][str(num)] = nn.Sequential(
                    encode_image_by_ntimes(0, ndf, layer_num),
                    nn.Conv2d(ndf * min(8, 2**(layer_num-1)), 1, kernel_size=1, stride=1),
                    nn.Sigmoid()
                )

        elif cfg.MODEL.COMP_NAME == 'none':
            self.img_code = nn.Conv2d(1, 1, 1) # useless


    def forward(self, x, size=None):
        if cfg.MODEL.COMP_NAME == 'pixel':
            x = self.img_code(x)
            x = self.conv(x)

        elif cfg.MODEL.COMP_NAME == 'glpuzzle':
            x = self.__dict__['_modules'][str(size)](x)

        return x