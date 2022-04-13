import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.Amodel import GlobalAttentionGeneral, AuxAttentionGeneral
from models.Bmodel import downBlock_G, GLU, upBlock, ResBlock_ADIN, conv1x1, conv3x3, PixelNorm, SPADE
from models.Cmodel import COLORG_NET_PIXEL, COLORG_NET_FOREBACK
from miscc.utils import combine_attn

from miscc.config import cfg

# ############## G networks ###################
class INIT_STAGE_G_PAT(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G_PAT, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM  # cfg.TEXT.EMBEDDING_DIM
        self.height = cfg.TREE.BASE_SIZE//8
        self.width = cfg.TREE.BASE_SIZE//8

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * self.height * self.width * 2, bias=False),
            nn.BatchNorm1d(ngf * self.height * self.width * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

    def forward(self, z_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/4 x 32 x 32
        """
        # state size ngf x 4 x 4
        out_code = self.fc(z_code)
        out_code = out_code.view(-1, self.gf_dim, self.height, self.width)
        # state size ngf/2 x 16 x 16
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 32 x 32
        out_code = self.upsample2(out_code)

        return out_code

class Z_MLP(nn.Module):
    def __init__(self, in_dim, n_mlp):
        super(Z_MLP, self).__init__()
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)
    def forward(self, x):
        return self.style(x)

class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = ncf  # cfg.TEXT.EMBEDDING_DIM

        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 8 * 2, bias=False),
            nn.BatchNorm1d(ngf * 8 * 8 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

    def forward(self, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/4 x 32 x 32
        """
        # state size ngf x 4 x 4
        out_code = self.fc(c_code)
        out_code = out_code.view(-1, self.gf_dim, 8, 8)
        # state size ngf/2 x 16 x 16
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 32 x 32
        out_code = self.upsample2(out_code)

        return out_code

class INIT_STAGE_G_SEG(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G_SEG, self).__init__()
        self.gf_dim = ngf # the base number of channels: e.g., 12
        self.in_dim = ncf # number of object parts of interest
        self.define_module()

    def define_module(self):
        ncf, ngf = self.in_dim, self.gf_dim
        # Convolution-InstanceNorm-ReLU
        self.conv3x3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ncf, ngf, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True))

        self.downsample1 = downBlock_G(ngf, ngf*2) # output 24 channels
        self.downsample2 = downBlock_G(ngf*2, ngf*4) # output 48 channels

    def forward(self, seg):
        """
        :param seg: batch x ncf x seg_size (128) x seg_size
        :return: batch x ngf*4 x 64 x 64
        """
        # state size ngf x 128 x 128
        out_code = self.conv3x3(seg)
        # state size ngf*2 x 64 x 64
        out_code = self.downsample1(out_code)
        # state size ngf*4 x 32 x 32
        out_code = self.downsample2(out_code)

        return out_code

class INIT_STAGE_G_MAIN(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G_MAIN, self).__init__()
        self.gf_dim = ngf # the base number of channels: e.g., 12
        self.define_module()

    def define_module(self):
        out_dim = self.gf_dim
        ngf = self.gf_dim
        self.layer_1 = ResBlock_ADIN(ngf)
        self.layer_2 = ResBlock_ADIN(ngf)
        self.upsample = upBlock(ngf, out_dim)


    def forward(self, z_code, seg, h_code1_att):
        """
        :param h_code_seg: batch x ngf x 32 x 32
        :param h_code_sent: batch x ngf x 32 x 32
        :return: batch x ngf*4 x 64 x 64
        """
        # state size ngf*2 x 32 x 32
        out_code = self.layer_1(h_code1_att, z_code, seg)
        out_code = self.layer_2(out_code, z_code, seg)

        # state size ngf x 64 x 64
        out_code = self.upsample(out_code)

        return out_code

class INIT_STAGE_G_ATT(nn.Module):
    def __init__(self, ngf, nef):
        super(INIT_STAGE_G_ATT, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.aux_att = AuxAttentionGeneral(self.gf_dim, self.ef_dim)
        self.att = GlobalAttentionGeneral(self.gf_dim, self.ef_dim)


    def forward(self, h_code, mask, ori_word_embs, aux_words_embs, uni_glove_words_embs, sem_seg, pooled_sem_seg):
        self.att.applyMask(mask)
        word_embs, vnl_att = self.att(h_code, ori_word_embs) ##########
        _, aux_att = self.aux_att(aux_words_embs, uni_glove_words_embs, pooled_sem_seg)
        c_code, new_att, vnl_att, aux_att = combine_attn(vnl_att, aux_att, word_embs, sem_seg, pooled_sem_seg)
        return c_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def define_module(self):
        ngf = self.gf_dim
        self.att = GlobalAttentionGeneral(ngf, self.ef_dim)
        self.aux_att = AuxAttentionGeneral(ngf, self.ef_dim)

        self.upsample = upBlock(ngf*2, self.gf_dim)
        self.layer_1 = ResBlock_ADIN(ngf*2)
        self.layer_2 = ResBlock_ADIN(ngf*2)


    def forward(self, z_code, h_code, ori_word_embs, mask, sem_seg, pooled_sem_seg, uni_glove_words_embs, aux_words_embs):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            sem_seg: batch x pncf x ih x iw
            pooled_sem_seg: batch x ih x iw
            aux_words_embs (aux_query): batch x cdf x ih x iw (queryL=ihxiw)
            uni_words_embs (aux_content): batch x cdf x sourceL (sourceL=seq_len)
            att1: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        word_embs, vnl_att = self.att(h_code, ori_word_embs) ##########
        _, aux_att = self.aux_att(aux_words_embs, uni_glove_words_embs, pooled_sem_seg)
        c_code, new_att, vnl_att, aux_att = combine_attn(vnl_att, aux_att, word_embs, sem_seg, pooled_sem_seg)

        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.layer_1(h_c_code, z_code, sem_seg)
        out_code = self.layer_2(out_code, z_code, sem_seg)

        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, new_att, vnl_att, aux_att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            self.z_net1_mlp = Z_MLP(cfg.GAN.Z_DIM, cfg.GAN.Z_MLP_NUM)
            self.h_net1_main = INIT_STAGE_G_MAIN(ngf)
            self.h_net1 = INIT_STAGE_G(ngf * 4, ncf)
            self.h_net1_att = INIT_STAGE_G_ATT(ngf, nef)
            self.img_net1 = GET_IMAGE_G(ngf)

        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net3 = GET_IMAGE_G(ngf)
            self.colorG = COLORG_NET_PIXEL()

    def forward(self, z_code, sent_emb, word_embs, mask,
        sem_segs, pooled_sem_segs, uni_glove_words_embs,
        aux_words_embs, imgs):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :param sem_segs: [batch x pncf x seg_size x seg_size]
            :param pooled_sem_segs: [batch x seg_size x seg_size]
            :param uni_words_embs: batch x cdf x seq_len
            :param aux_words_embs: [batch x idf x seg_size x seg_size]
            :return:
        """
        raw_fake_imgs = []
        att_maps, vnl_att_maps, aux_att_maps = [], [], []
        sent_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 0:
            z_code = self.z_net1_mlp(z_code)
            height_half, width_half = cfg.TREE.BASE_SIZE//2, cfg.TREE.BASE_SIZE//2
            seg_half = F.interpolate(sem_segs[0], (height_half, width_half), mode='nearest')
            pooled_seg_half = torch.max(seg_half, dim=1)[0]
            aux_words_half = F.interpolate(aux_words_embs[0], (height_half, width_half), mode='nearest')
            h_code1_sent = self.h_net1(sent_code)

            h_code1_word = self.h_net1_att(h_code1_sent, mask, word_embs, aux_words_half, uni_glove_words_embs, seg_half, pooled_seg_half)

            h_code1 = self.h_net1_main(z_code, seg_half, h_code1_word)
            raw_fake_img1 = self.img_net1(h_code1)
            raw_fake_imgs.append(raw_fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1, vnl_att1, aux_att1 = self.h_net2(z_code, h_code1,
                word_embs, mask, sem_segs[0], pooled_sem_segs[0],
                uni_glove_words_embs, aux_words_embs[0])
            raw_fake_img2 = self.img_net2(h_code2)
            raw_fake_imgs.append(raw_fake_img2)
            if att1 is not None:
                att_maps.append(att1)
                vnl_att_maps.append(vnl_att1)
                aux_att_maps.append(aux_att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2, vnl_att2, aux_att2 = self.h_net3(z_code, h_code2,
                word_embs, mask, sem_segs[1], pooled_sem_segs[1],
                uni_glove_words_embs, aux_words_embs[1])
            raw_fake_img3 = self.img_net3(h_code3)
            raw_fake_imgs.append(raw_fake_img3)
            # background
            back_index = (pooled_sem_segs[2] == 0).unsqueeze(dim=1).expand_as(imgs[2])
            front_index = (pooled_sem_segs[2] == 1).unsqueeze(dim=1).expand_as(imgs[2])
            raw_fake_img_background = torch.zeros(raw_fake_img3.shape).cuda()
            raw_fake_img_background[back_index] = imgs[2][back_index]
            raw_fake_img_background[front_index] = raw_fake_img3[front_index]  # [batch_size, channel, height, width]
            fake_img = self.colorG(raw_fake_img_background)
            if att2 is not None:
                att_maps.append(att2)
                vnl_att_maps.append(vnl_att2)
                aux_att_maps.append(aux_att2)

        return raw_fake_imgs, fake_img, att_maps, vnl_att_maps, aux_att_maps, mu, logvar