import os
import sys
import errno
import math
import numpy as np
from scipy import linalg
from torch.nn import init
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform
import random

from miscc.config import cfg


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50

# fnt = ImageFont.truetype('E:/new_guided_attentioin/birds/font/times.ttf', 30)
fnt = None

def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM,
                       nvis=cfg.TRAIN.NVIS):
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)
    text_convas = np.ones([batch_size * FONT_MAX, (max_word_num + 2) * (vis_size + 2), 3], dtype=np.uint8)
    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    real_imgs = nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size//att_sze)>1:
                one_map = F.interpolate(torch.from_numpy(one_map).permute((2,0,1)).unsqueeze(dim=0), (vis_size, vis_size)).squeeze(dim=0).permute((1,2,0)).numpy()
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None

def build_action_images(real_imgs, data_dir, key, fullpath, image_transform, frame_num, captions, ixtoword):
    action_path = '%s/CUB_200_2011/action_shape/%s/' % (data_dir, key.split('/')[0])
    sem_imgs = []
    sem_name_list = os.listdir(action_path)
    for sem_name in sem_name_list:
        if sem_name.rfind('.png') != -1:
            sem_img_path = os.path.join(action_path, sem_name)
            sem_img = Image.open(sem_img_path)
            sem_img = image_transform(sem_img)
            sem_imgs.append(sem_img)
    width, height = real_imgs[0].size
    assert width == height
    black_height = 40
    black_width = 100
    merged = Image.new('RGBA', (width * frame_num + black_width, height * 2 + black_height), (0, 0, 0, 0))
    img_left = 0
    for i in range(frame_num):
        merged.paste(real_imgs[i], (img_left, black_height))
        merged.paste(sem_imgs[i], (img_left, black_height + height))
        img_left += width
    sentence = ''
    for cap in captions:
        cap = int(cap.cpu().numpy())
        cap = ixtoword[cap]
        if cap == '<end>': break
        sentence += cap + ' '
    draw = ImageDraw.Draw(merged)
    draw.text((0,0), sentence, font=fnt, fill=(0, 0, 0, 255))
    merged.save(fullpath)

####################################################################
class CKPT:
    def __init__(self, epoch, fid):
        self.epoch = epoch
        self.fid = fid

    def get_name(self):
        return 'netG_epoch_%d_%.2f.pth' % (self.epoch, self.fid)

    def get_epoch(self):
        return self.epoch

    def get_fid(self):
        return self.fid

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_fid(self, fid):
        self.fid = fid

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def geo_shuffle(sem_segs, batch_size):
    swap_index = torch.randperm(batch_size)
    new_segs = []
    for i in range(cfg.TREE.BRANCH_NUM):
        new_sem = torch.zeros(sem_segs[i].shape).cuda()
        new_sem[swap_index] = sem_segs[i][swap_index]
        new_segs.append(new_sem)
    return new_segs

def form_uni_cap_batch(captions, cap_lens):
    cap_lens = cap_lens.data.cpu().numpy().tolist()
    uni_captions = []
    for i in range(len(cap_lens)):
        uni_captions.append(captions[i,:cap_lens[i]])
    uni_captions = torch.cat(uni_captions)
    uni_captions = uni_captions.unsqueeze(1)
    uni_cap_lens = torch.ones(uni_captions.size(0)).long()
    uni_cap_lens = Variable(uni_cap_lens).cuda()
    return uni_captions, uni_cap_lens

def reorg_uni_caps(uni_words_emb, cap_lens):
    cap_lens = cap_lens.data.cpu().numpy().tolist()
    batch_size, max_len = len(cap_lens), max(cap_lens)
    emb_dim = uni_words_emb.size(1)
    new_uni_words_emb = torch.zeros(batch_size, max_len, emb_dim)
    new_uni_words_emb = Variable(new_uni_words_emb).cuda()
    tmp_num = 0
    for i in range(len(cap_lens)):
        new_uni_words_emb[i,:cap_lens[i],:] = uni_words_emb[tmp_num:tmp_num+cap_lens[i]]
        tmp_num += cap_lens[i]
    new_uni_words_emb = torch.transpose(new_uni_words_emb, 1, 2)
    return new_uni_words_emb

def form_aux_words_embs(plabels_emb, sem_segs):
    # plabels_emb: cfg.GAN.P_NUM x cfg.TEXT.GLOVE_EMBEDDING_DIM
    # sem_segs: [batch_size x cfg.GAN.P_NUM x seg_size x seg_size]
    # aux_words_embs: [batch_size x cfg.TEXT.GLOVE_EMBEDDING_DIM x seg_size x seg_size]
    # return aux_words_embs
    aux_words_embs = []
    batch_size = sem_segs[0].size(0)
    for sem_seg in sem_segs:
        seg_size = sem_seg.size(2)
        sem_seg = torch.transpose(sem_seg, 0, 1).contiguous()
        sem_seg = sem_seg.view(cfg.GAN.P_NUM, -1)

        aux_words_emb = torch.zeros(batch_size*seg_size*seg_size, cfg.TEXT.GLOVE_EMBEDDING_DIM)
        aux_words_emb = Variable(aux_words_emb).cuda()
        for i in range(cfg.GAN.P_NUM):
            indices = (sem_seg[i] == 1).nonzero()
            if indices.nelement() == 0:
                continue
            if len(indices) > 1:
                indices = indices.squeeze()

            aux_words_emb[indices] = plabels_emb[i].repeat(len(indices), 1)

        aux_words_emb = aux_words_emb.view(batch_size, seg_size, seg_size, -1)
        aux_words_emb = aux_words_emb.permute(0,3,1,2)
        aux_words_embs.append(aux_words_emb)
    
    return aux_words_embs

def combine_attn(attn, aux_attn, sourceT, sem_seg, pooled_sem_seg):
    # attn: batch_size x sourceL x queryL (sourceL=seq_len, queryL=ihxiw)
    # aux_attn: batch_size x sourceL x queryL (sourceL=seq_len, queryL=ihxiw)
    # sourceT: batch_size x idf x sourceL
    # sem_seg: batch_size x cfg.GAN.P_NUM x seg_size x seg_size
    batch_size, seg_size = sem_seg.size(0), sem_seg.size(2)
    sourceL = attn.size(1)

    # -> cfg.GAN.P_NUM x batch_size*seg_size*seg_size
    sem_seg = torch.transpose(sem_seg, 0, 1).contiguous()
    sem_seg = sem_seg.view(cfg.GAN.P_NUM, -1)

    # -> batch_size*seg_size*seg_size
    pooled_sem_seg = pooled_sem_seg.view(-1)

    # -> sourceL x batch_size*seg_size*seg_size
    attn = torch.transpose(attn, 0, 1).contiguous()
    attn = attn.view(sourceL, -1)

    aux_attn = torch.transpose(aux_attn, 0, 1).contiguous()
    aux_attn = aux_attn.view(sourceL, -1)

    new_attn = torch.zeros(attn.size())
    new_attn = Variable(new_attn).cuda()

    for i in range(cfg.GAN.P_NUM):
        nonzero_indices = (sem_seg[i] == 1).nonzero()
        if nonzero_indices.nelement() > 0:
            if len(nonzero_indices) > 1:
                nonzero_indices = nonzero_indices.squeeze()
            new_attn[:, nonzero_indices] = torch.max(attn[:,nonzero_indices], aux_attn[:,nonzero_indices])


    zero_indices = (pooled_sem_seg == 0).nonzero()
    if zero_indices.nelement() > 0:
        if len(zero_indices) > 1:
            zero_indices = zero_indices.squeeze()
        new_attn[:, zero_indices] = attn[:,zero_indices]*cfg.TRAIN.SMOOTH.ALPHA

    attn = attn.view(sourceL, batch_size, -1)
    attn = torch.transpose(attn, 0, 1).contiguous()
    attn = attn.view(batch_size, -1, seg_size, seg_size)

    aux_attn = aux_attn.view(sourceL, batch_size, -1)
    aux_attn = torch.transpose(aux_attn, 0, 1).contiguous()
    aux_attn = aux_attn.view(batch_size, -1, seg_size, seg_size)

    new_attn = new_attn.view(sourceL, batch_size, -1)
    new_attn = torch.transpose(new_attn, 0, 1).contiguous()
    # (batch x idf x sourceL)(batch x sourceL x queryL)
    # --> batch x idf x queryL
    weightedContext = torch.bmm(sourceT, new_attn)
    weightedContext = weightedContext.view(batch_size, -1, seg_size, seg_size)
    new_attn = new_attn.view(batch_size, -1, seg_size, seg_size)
    return weightedContext, new_attn, attn, aux_attn

def check_nopart(seg):
    batch = seg[-1].shape[0]
    for i in range(batch):
        if torch.nonzero(seg[-1][i]).shape[0] == 0:
            return False
    return True

def random_img(img, height, width, patch_height, patch_width, pn):
    real_patch_list = []
    for i in range(pn):
        random_height = random.randint(0, height - patch_height)
        random_width = random.randint(0, width - patch_width)
        real_patch = img[:, :, random_height:random_height + patch_height, random_width:random_width + patch_width].unsqueeze(dim=0)
        real_patch_list.append(real_patch)
    return real_patch_list

def random_fake_batch(fake_img_background, pool_seg, patch_height, patch_width, batch):
    fake_patch_list = []
    height, width = pool_seg[0].shape
    for i in range(batch):
        index_one = torch.nonzero(pool_seg[i])
        index = random.randint(0, index_one.shape[0] - 1)
        height_cen, width_cen = index_one[index]
        random_up = height_cen - patch_height//2
        random_down = height_cen + (patch_height - patch_height//2)
        random_left = width_cen - patch_width//2
        random_right = width_cen + (patch_width - patch_width//2)
        if random_up>=0 and random_down<height:
            random_height = random_up
        elif random_up<0:
            random_height = 0
        elif random_down>=height:
            random_height = height - patch_height
        if random_left>=0 and random_right<width:
            random_width = random_left
        elif random_left<0:
            random_width = 0
        elif random_right>=width:
            random_width = width - patch_width
        fake_patch = fake_img_background[i:i + 1, :, random_height:random_height + patch_height, random_width:random_width + patch_width]
        fake_patch_list.append(fake_patch)
    fake_patches = torch.cat(fake_patch_list, dim=0).unsqueeze(dim=0)
    return fake_patches

def random_fake(fake_img_background, pool_seg, patch_height, patch_width, pn, batch):
    fake_patch_list = []
    for i in range(pn):
        fake_patches = random_fake_batch(fake_img_background, pool_seg, patch_height, patch_width, batch)
        fake_patch_list.append(fake_patches)
    return fake_patch_list

def build_super_patch(gt_tensor, patches, class_num, patch_height, patch_width, batch):
    base_size = int(math.sqrt(class_num))
    super_gt = torch.zeros([batch, base_size, base_size]).cuda()
    super_patch = torch.zeros([batch, 3, base_size * patch_height, base_size * patch_width])
    pn_bit = 0
    height_i = 0
    for i in range(base_size):
        width_j = 0
        for j in range(base_size):
            super_gt[:, i, j] = gt_tensor[:, pn_bit]
            super_patch[:, :, height_i:height_i+patch_height, width_j:width_j+patch_width] = patches[:, pn_bit, :, :, :]
            width_j += patch_width
            pn_bit += 1
        height_i += patch_height
    return super_gt, super_patch

def visual_pixel(fake_groundtruth, fake_seg_result, fake_img_background):
    fake_groundtruth = fake_groundtruth[0,0,:,:]
    fake_seg_result = fake_seg_result[0,0,:,:]
    fake_img_background = (fake_img_background[0].detach().cpu().permute((1,2,0)) + 1) *127.5
    fake_gt = Image.fromarray(np.uint8(fake_groundtruth.detach().cpu().numpy()*255))
    fake_rt = Image.fromarray(np.uint8(fake_seg_result.detach().cpu().numpy()*255))
    fake_img = Image.fromarray(np.uint8(fake_img_background))
    fake_gt.save('fake_gt.png')
    fake_rt.save('fake_rt.png')
    fake_img.save('img.png')
    exit()

def visual_glpuzzle(super_patch, super_gt, fake_gt, pre_gt, batch, epoch, pn, image_dir):
    from PIL import Image
    visual_img = Image.new('RGB', (256 * 4, 256 * batch), (0, 0, 0, 0))
    for i in range(batch):
        super_patch_np = (super_patch[i] + 1) * 127.5
        super_patch_img = Image.fromarray(np.uint8(super_patch_np.permute((1, 2, 0)).detach().cpu()))
        super_gt_img = Image.fromarray(np.uint8((super_gt[i] * 255).detach().cpu())).resize((256, 256))
        fake_gt_img = Image.fromarray(np.uint8((fake_gt[i] * 255).detach().cpu())).resize((256,256))
        pre_gt_img = Image.fromarray(np.uint8((pre_gt[i] * 255).detach().cpu())).resize((256,256))
        visual_img.paste(super_patch_img, (256 * 0, 256 * i))
        visual_img.paste(super_gt_img, (256 * 1, 256 * i))
        visual_img.paste(fake_gt_img, (256 * 2, 256 * i))
        visual_img.paste(pre_gt_img, (256 * 3, 256 * i))
    save_pth = os.path.join(image_dir, '{}_{}_visual.png'.format(epoch, pn))
    if not os.path.exists(save_pth):
        visual_img.save(save_pth)

def match_cap(cap, ixtoword, glove_cap, glove_ixtoword):
    def wrong():
        print('captions do not match!')
        exit()
    def right():
        print('captions match!')
    for i in range(len(cap)):
        # print(i, len(cap[i]))
        # for j in range(len(cap[i])): print(ixtoword[cap[i][j]],end = ' ')
        # print('')
        # for j in range(len(glove_cap[i])): print(glove_ixtoword[glove_cap[i][j]], end=' ')
        # print('')
        if len(cap[i]) != len(glove_cap[i]): wrong()
        for j in range(len(cap[i])):
            # print(cap[i][j], glove_cap[i][j],ixtoword[cap[i][j]],glove_ixtoword[glove_cap[i][j]])
            if ixtoword[cap[i][j]] != glove_ixtoword[glove_cap[i][j]]:
                # print(ixtoword[cap[i][j]],glove_ixtoword[glove_cap[i][j]])
                wrong()
    right()

def get_activations(images, model, batch_size, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    #d0 = images.shape[0]
    d0 = int(images.size(0))
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, cfg.TEST.FID_DIMS))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        '''batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cfg.CUDA:
            batch = batch.cuda()'''
        batch = images[start:end]

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act      : Numpy array of dimension (n_images, dim (e.g. 2048)).

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_inception_score(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)