import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import random
import math
from miscc.config import cfg
from miscc.utils import l1norm, l2norm
from miscc.utils import random_img, random_fake, build_super_patch


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))

def func_attn(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)

def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size, keys, top1=True):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            if isinstance(class_ids, np.ndarray):
                mask = (class_ids == class_ids[i]).astype(np.uint8)
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
            else:
                mask = np.zeros([batch_size]).astype(np.uint8)
                part_name = keys[i].split('/')[0]
                for j in range(cfg.GAN.MAX_LENGTH_CLS):
                    if class_ids[i][j].split('_')[0] == part_name:
                        for k in range(batch_size):
                            if class_ids[i][j] in class_ids[k]:
                                mask[k] = 1
                        break
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))

        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attn(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks).cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks.byte(), -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        if top1:
            _, predicted = torch.max(similarities, 1)
            _, predicted1 = torch.max(similarities1, 1)
            correct = ((predicted == labels).sum().cpu().item() +
                       (predicted1 == labels).sum().cpu().item())
            accuracy = (100. * correct) / (batch_size * 2.)
            # print('w_correct = ', correct, 'w_accuracy = ', accuracy)
        else:
            # similarities(i, j): the similarity between the i-th image and the j-th text
            # similarities1(i, j): the similarity between the i-th text and the j-th image
            # accuracy = [similarities, similarities1]
            accuracy = [nn.Softmax()(similarities), nn.Softmax()(similarities1)]
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps, accuracy

def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, keys, eps=1e-8, top1=True):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            if isinstance(class_ids, np.ndarray):
                mask = (class_ids == class_ids[i]).astype(np.uint8)
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
            else:
                mask = np.zeros([batch_size]).astype(np.uint8)
                part_name = keys[i].split('/')[0]
                for j in range(cfg.GAN.MAX_LENGTH_CLS):
                    if class_ids[i][j].split('_')[0] == part_name:
                        for k in range(batch_size):
                            if class_ids[i][j] in class_ids[k]:
                                mask[k] = 1
                        break
                mask[i] = 0
                masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks).cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks.byte(), -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
        if top1:
            _, predicted0 = torch.max(scores0, 1)
            _, predicted1 = torch.max(scores1, 1)
            correct = ((predicted0 == labels).sum().cpu().item() +
                       (predicted1 == labels).sum().cpu().item())
            accuracy = (100. * correct) / (batch_size * 2.)
        else:
            # s0(i, j): the similarity between the i-th image and the j-th text
            # s1(i, j): the similarity between the i-th text and the j-th image
            # accuracy = [scores0, scores1]
            # accuracy = [nn.Softmax()(scores0), nn.Softmax()(scores1)]
            accuracy = scores0
        # print('s_correct = ', correct, 's_accuracy = ', accuracy)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, accuracy


# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    cond_real_logits = netD.module.COND_DNET(real_features, conditions)
    cond_fake_logits = netD.module.COND_DNET(fake_features, conditions)
    real_labels = Variable(torch.FloatTensor(cond_real_logits.size()).fill_(1)).cuda()
    fake_labels = Variable(torch.FloatTensor(cond_fake_logits.size()).fill_(0)).cuda()
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.module.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    real_logits = netD.module.UNCOND_DNET(real_features)
    fake_logits = netD.module.UNCOND_DNET(fake_features)
    real_errD = nn.BCELoss()(real_logits, real_labels)
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
    errD = (real_errD / 2. + fake_errD / 3.) * cfg.TRAIN.SMOOTH.LAMBDA2 +\
           (cond_real_errD / 2. + cond_fake_errD / 3. + cond_wrong_errD / 3.)
    return errD

def discriminator_shape_loss(segD, real_imgs, fake_imgs, sem_segs):
    #shape forward
    real_features_seg = segD(real_imgs, sem_segs)
    fake_features_seg = segD(fake_imgs.detach(), sem_segs)
    #shape error
    real_logits_seg = segD.module.UNCOND_DNET(real_features_seg)
    fake_logits_seg = segD.module.UNCOND_DNET(fake_features_seg)
    real_labels = Variable(torch.FloatTensor(real_logits_seg.size()).fill_(1)).cuda()
    fake_labels = Variable(torch.FloatTensor(fake_logits_seg.size()).fill_(0)).cuda()
    real_seg_loss = nn.BCELoss()(real_logits_seg, real_labels)
    fake_seg_loss = nn.BCELoss()(fake_logits_seg, fake_labels)
    #shape wrong loss
    batch_size = real_features_seg.size(0)
    seg_wrong_logits = segD.module.UNCOND_DNET(real_features_seg[:(batch_size - 1)], sem_segs[1:batch_size])
    wrong_errD_seg = nn.BCELoss()(seg_wrong_logits, fake_labels[1:batch_size])
    errD_seg = real_seg_loss + fake_seg_loss + wrong_errD_seg
    return errD_seg

def segD_loss(netSegD, fake_img_background, pool_seg):
    fake_seg_result = netSegD(fake_img_background.detach())
    seg_groundtruth = pool_seg.unsqueeze(dim=1)
    errD = nn.BCELoss()(fake_seg_result, seg_groundtruth)

    # tfrt score
    pre_gt = torch.zeros(seg_groundtruth.shape).cuda()
    pre_gt[fake_seg_result > 0.5] = 1
    front_index = (seg_groundtruth == 1)
    back_index = (seg_groundtruth == 0)
    front_right_num = torch.nonzero(pre_gt[front_index] == seg_groundtruth[front_index]).shape[0]
    front_wrong_num = torch.nonzero(pre_gt[front_index] != seg_groundtruth[front_index]).shape[0]
    back_right_num = torch.nonzero(pre_gt[back_index] == seg_groundtruth[back_index]).shape[0]
    back_wrong_num = torch.nonzero(pre_gt[back_index] != seg_groundtruth[back_index]).shape[0]
    # min max
    front_wrong_max = back_wrong_max = -999
    front_wrong_min = back_wrong_min = 999
    front_wrong_index = (pre_gt[front_index] != seg_groundtruth[front_index])
    if True in front_wrong_index:
        front_wrong_max = torch.max(fake_seg_result[front_index][front_wrong_index])
        front_wrong_min = torch.min(fake_seg_result[front_index][front_wrong_index])
    back_wrong_index = (pre_gt[back_index] != seg_groundtruth[back_index])
    if True in back_wrong_index:
        back_wrong_max = torch.max(fake_seg_result[back_index][back_wrong_index])
        back_wrong_min = torch.min(fake_seg_result[back_index][back_wrong_index])
    return errD, (front_right_num, front_wrong_num, front_wrong_max, front_wrong_min, back_right_num, back_wrong_num, back_wrong_max, back_wrong_min)

def segD_glpuzzle_loss(netSegD, fake_img_background, img, pool_seg, choice, epoch=None, image_dir=None):
    batch = fake_img_background.shape[0]
    puzzle_num = cfg.GAN.PUZZLE_NUM
    height = fake_img_background.shape[2]
    width = fake_img_background.shape[3]
    puzzle_loss = 0
    total_right, total_wrong = 0, 0
    for pn in puzzle_num:
        patch_height = height // pn
        patch_width = width // pn
        tfrt = cfg.MODEL.TFRT
        real_patch_list = random_img(img, height, width, patch_height, patch_width, (pn * pn) // (tfrt + 1) * tfrt)

        fake_patch_list = random_fake(fake_img_background, pool_seg, patch_height, patch_width, (pn * pn) // (tfrt + 1), batch)
        real_patch_list.extend(fake_patch_list)
        patches = torch.cat(real_patch_list, dim=0) #[real,real,...,fake,fake] length = pn*pn  [p, b, c, h, w]
        patches = patches.permute((1, 0, 2, 3, 4)).contiguous() #[batch, patch, channel, height, width]
        class_num = patches.shape[1]
        false_start_index = class_num // (cfg.MODEL.TFRT + 1) * cfg.MODEL.TFRT
        gt_list = []
        for i in range(batch):
            index = torch.randperm(class_num)
            patches[i] = patches[i][index]
            gt_tensor_i = torch.zeros(class_num).cuda()
            gt_tensor_i[index >= false_start_index] = 1
            gt_list.append(gt_tensor_i.unsqueeze(dim=0))
        gt_tensor = torch.cat(gt_list, dim=0) #[batch, pn*pn]
        super_gt, super_patch = build_super_patch(gt_tensor, patches, class_num, patch_height, patch_width, batch)
        if choice == 'test':
            patch_result = netSegD(super_patch.detach(), pn).squeeze(dim=1)
            puzzle_loss += nn.BCELoss()(patch_result, super_gt)

        elif choice == 'train':
            patch_result = netSegD(super_patch, pn).squeeze(dim=1)
            pre_gt = torch.zeros(super_gt.shape).cuda()
            pre_gt[patch_result > 0.5] = 1
            fake_gt = torch.zeros(super_gt.shape).cuda()
            fake_gt[(super_gt - pre_gt) == -1] = 1
            from miscc.utils import visual_glpuzzle
            visual_glpuzzle(super_patch, super_gt, fake_gt, patch_result, batch, epoch, pn, image_dir)
            puzzle_loss += nn.BCELoss()(patch_result, fake_gt)

    if choice == 'train':
        return puzzle_loss
    elif choice == 'test':
        return puzzle_loss

def generator_loss(netsD, shpsD, segD, image_encoder, raw_fake_imgs, fake_img_background, sem_segs, pooled_sem_segs, words_embs, sent_emb, match_labels, cap_lens, class_ids, imgs, epoch, image_dir, keys):
    numDs = len(netsD)
    batch_size = raw_fake_imgs[0].size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](raw_fake_imgs[i])
        cond_logits = netsD[i].module.COND_DNET(features, sent_emb)
        real_labels = Variable(torch.FloatTensor(cond_logits.size()).fill_(1)).cuda()
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        logits = netsD[i].module.UNCOND_DNET(features)
        errG = nn.BCELoss()(logits, real_labels)
        g_loss = errG * cfg.TRAIN.SMOOTH.LAMBDA2 + cond_errG
        if i == numDs - 1:
            features = netsD[i](fake_img_background)
            cond_logits = netsD[i].module.COND_DNET(features, sent_emb)
            real_labels = Variable(torch.FloatTensor(cond_logits.size()).fill_(1)).cuda()
            cond_errG = nn.BCELoss()(cond_logits, real_labels)
            logits = netsD[i].module.UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss += errG * cfg.TRAIN.SMOOTH.LAMBDA2 + cond_errG

        errG_total += g_loss
        logs += 'g_loss%d: %.2f ' % (i, g_loss.item())
        ### shape_loss
        features_seg = shpsD[i](raw_fake_imgs[i], sem_segs[i])
        logits_seg = shpsD[i].module.UNCOND_DNET(features_seg)
        shape_loss = nn.BCELoss()(logits_seg, real_labels)
        errG_total += shape_loss
        logs += 'shape_loss%d: %.2f ' % (i, shape_loss.item())
        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(raw_fake_imgs[i])
            w_loss0, w_loss1, _, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size, keys)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            s_loss0, s_loss1, _ = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size, keys)
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())

            segG_loss = segD_glpuzzle_loss(segD, fake_img_background, imgs[-1], pooled_sem_segs[-1], 'train', epoch, image_dir)
            errG_total += cfg.TRAIN.SMOOTH.LAMBDA3 * segG_loss
            logs += 'errSegG%d: %.2f ' % (i, segG_loss.item())

            front_index = (pooled_sem_segs[i] == 1).unsqueeze(dim=1).expand_as(fake_img_background)
            color_l1_loss = (fake_img_background[front_index] - raw_fake_imgs[i][front_index]).abs().sum() / front_index.sum()
            errG_total += cfg.TRAIN.SMOOTH.LAMBDA4 * color_l1_loss
            logs += 'colorG_loss%d: %.2f ' % (i, color_l1_loss.item())
    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
