from __future__ import print_function
from six.moves import range
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, CKPT
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.utils import form_aux_words_embs, check_nopart
from models.Dmodel import SEG_D_NET
from models.Emodel import RNN_ENCODER, CNN_ENCODER
from models.Pmodel import INCEPTION_V3, INCEPTION_V3_FID
from models.Rmodel import G_NET
from miscc.losses import discriminator_loss, generator_loss, KL_loss, words_loss, sent_loss, segD_loss, segD_glpuzzle_loss
from miscc.losses import discriminator_shape_loss
from Datasets.datasetTrain import prepare_train_data
from Datasets.datasetTest import prepare_test_data
from miscc.utils import get_activations, compute_inception_score, negative_log_posterior_probability
from miscc.utils import calculate_activation_statistics, calculate_frechet_distance

import os
import time

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, dataset, test_dataloader, test_dataset, args):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.test_batch_size = cfg.TEST.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.n_words = dataset.n_words
        self.ixtoword = dataset.ixtoword
        self.part_labels = dataset.part_labels
        self.part_label_lens = dataset.part_label_lens
        self.sorted_part_label_indices = dataset.sorted_part_label_indices
        self.data_loader = data_loader
        self.test_data_loader = test_dataloader
        self.num_batches = len(self.data_loader)
        self.args = args
        # metric
        self.inception_model = INCEPTION_V3()
        self.inception_model.cuda()
        self.inception_model = nn.DataParallel(self.inception_model)
        self.inception_model.eval()
        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[cfg.TEST.FID_DIMS]
        self.inception_model_fid = INCEPTION_V3_FID([block_idx])
        self.inception_model_fid.cuda()
        self.inception_model_fid = nn.DataParallel(self.inception_model_fid)
        self.inception_model_fid.eval()
        self.test_dataset.create_acts()
        self.min_fid=999
        self.min_epoch=-1
        self.ckpt_lst = []


    def build_models(self, states):
        # ###################encoders######################################## #
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = os.path.join(cfg.PRETRAINED_DIR, 'image_encoder200.pth')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()
        image_encoder = image_encoder.cuda()
        image_encoder = nn.DataParallel(image_encoder, device_ids=cfg.GPU_group)

        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        txt_encoder_path = os.path.join(cfg.PRETRAINED_DIR, 'text_encoder200.pth')
        state_dict = torch.load(txt_encoder_path, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', txt_encoder_path)
        text_encoder.eval()
        text_encoder = text_encoder.cuda()
        # text_encoder = nn.DataParallel(text_encoder, device_ids=cfg.GPU_group)

        # #######################generator and discriminators############## #
        netsD = []
        shpsD = []
        from models.Dmodel import PAT_D_NET64, PAT_D_NET128, PAT_D_NET256
        from models.Dmodel import SHP_D_NET64, SHP_D_NET128, SHP_D_NET256
        netG = nn.DataParallel(G_NET(),device_ids=cfg.GPU_group)
        segD = nn.DataParallel(SEG_D_NET(), device_ids=cfg.GPU_group)
        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(nn.DataParallel(PAT_D_NET64(),device_ids=cfg.GPU_group))
            shpsD.append(nn.DataParallel(SHP_D_NET64(3), device_ids=cfg.GPU_group))
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(nn.DataParallel(PAT_D_NET128(),device_ids=cfg.GPU_group))
            shpsD.append(nn.DataParallel(SHP_D_NET128(3),device_ids=cfg.GPU_group))
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(nn.DataParallel(PAT_D_NET256(),device_ids=cfg.GPU_group))
            shpsD.append(nn.DataParallel(SHP_D_NET256(3),device_ids=cfg.GPU_group))
        netG.apply(weights_init)
        segD.apply(weights_init)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            shpsD[i].apply(weights_init)
        print('# of netsD', len(netsD))
        epoch = 0
        if states is not None:
            epoch = int(states['epoch'])+1
            netG.load_state_dict(states['netG'])
            segD.load_state_dict(states['netSegD'])
            for i in range(len(netsD)):
                netsD[i].load_state_dict(states['netPatD{}'.format(i)])
                shpsD[i].load_state_dict(states['netShpD{}'.format(i)])
        netG.cuda()
        segD.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
            shpsD[i].cuda()

        return [text_encoder, image_encoder, netG, netsD, shpsD, segD, epoch]

    def define_optimizers(self, netG, netsD, shpsD, segD, states):
        optimizersD = []
        optimizerShpsD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)
            optSegD = optim.Adam(shpsD[i].parameters(),
                                 lr=cfg.TRAIN.DISCRIMINATOR_LR,
                                 betas=(0.5, 0.999))
            optimizerShpsD.append(optSegD)

        optimizerSegD = optim.Adam(segD.parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        if states is not None:
            optimizerG.load_state_dict(states['optimizerG'])
            optimizerSegD.load_state_dict(states['optimizerSegD'])
            for i in range(len(netsD)):
                optimizersD[i].load_state_dict(states['optimizerPatD{}'.format(i)])
                optimizerShpsD[i].load_state_dict(states['optimizerShpD{}'.format(i)])

        return optimizerG, optimizersD, optimizerShpsD, optimizerSegD

    def prepare_labels(self, train=True):
        if train:
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size
        match_labels = Variable(torch.LongTensor(range(batch_size))).cuda()
        return match_labels

    def register_ckpt(self, epoch, curfid):
        ckpt_ind = None
        if len(self.ckpt_lst) < cfg.TRAIN.CKPT_UPLIMIT:
            ckpt_obj = CKPT(epoch, curfid)
            self.ckpt_lst.append(ckpt_obj)
            ckpt_ind = len(self.ckpt_lst) - 1
        else:
            ind_w_maxfid = None
            ind_count = 0
            maxfid = -1
            for ckpt_obj_ in self.ckpt_lst:
                if ckpt_obj_.fid > maxfid:
                    ind_w_maxfid = ind_count
                    maxfid = ckpt_obj_.fid
                ind_count += 1
            if curfid < maxfid:
                name_w_maxfid = self.ckpt_lst[ind_w_maxfid].get_name()
                path_w_maxfid = os.path.join(self.model_dir, name_w_maxfid)
                os.remove(path_w_maxfid)
                rm_epoch = self.ckpt_lst[ind_w_maxfid].get_epoch()
                os.remove(os.path.join(self.image_dir, 'G_average_{}.png'.format(rm_epoch)))
                os.remove(os.path.join(self.image_dir, 'G_average_{}_aux.png'.format(rm_epoch)))
                for pn in cfg.GAN.PUZZLE_NUM:
                    visual_pth = os.path.join(self.image_dir, '{}_{}_visual.png'.format(rm_epoch, pn))
                    if os.path.exists(visual_pth):
                        os.remove(visual_pth)
                self.ckpt_lst[ind_w_maxfid].set_epoch(epoch)
                self.ckpt_lst[ind_w_maxfid].set_fid(curfid)
                ckpt_ind = ind_w_maxfid

        return ckpt_ind

    def save_model(self, netG, avg_param_G, netsD, shpsD, segD, optimizerG, optimizersShpD, optimizersPatD, optimizerSegD, epoch, fid, ckpt_ind):
        states = {'epoch': epoch, 'fid_score': fid, 'netG': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                  'netSegD': segD.state_dict(), 'optimizerSegD': optimizerSegD.state_dict()}

        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        states['avg_netG'] = netG.state_dict()

        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            states['netPatD{}'.format(i)] = netsD[i].state_dict()
            states['optimizerPatD{}'.format(i)] = optimizersPatD[i].state_dict()
            states['netShpD{}'.format(i)] = shpsD[i].state_dict()
            states['optimizerShpD{}'.format(i)] = optimizersShpD[i].state_dict()

        ckpt_name = self.ckpt_lst[ckpt_ind].get_name()
        ckpt_path = os.path.join(self.model_dir, ckpt_name)
        torch.save(states, ckpt_path)
        print('Save G/Ds models.')

    def save_img_results(self, imgs, netG, noise, sent_emb, words_embs, mask,
            captions, gen_iterations, sem_segs, pooled_sem_segs,
            uni_glove_words_embs, aux_words_embs, name='current'):
        # Save images
        raw_fake_imgs, fake_img_front, att_maps, vnl_att_maps, aux_att_maps, _, _ = netG(noise, sent_emb, words_embs, mask, sem_segs, pooled_sem_segs, uni_glove_words_embs, aux_words_embs, imgs)
        back_index = (pooled_sem_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
        front_index = (pooled_sem_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
        fake_img_background = torch.zeros(fake_img_front.shape)
        fake_img_background[back_index] = imgs[-1][back_index].cpu().detach()
        fake_img_background[front_index] = fake_img_front[front_index].cpu().detach()  # [batch_size, channel, height, width]
        attn_maps = att_maps[-1]
        att_sze = attn_maps.size(2)
        img_set, _ = build_super_images(fake_img_background, captions, self.ixtoword, attn_maps, att_sze, lr_imgs=raw_fake_imgs[-1].cpu().detach())
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/G_%s_%d.png' % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

        aux_attn_maps = aux_att_maps[-1]
        att_sze = aux_attn_maps.size(2)
        img_set, _ = build_super_images(fake_img_background, captions, self.ixtoword, aux_attn_maps, att_sze, lr_imgs=raw_fake_imgs[-1].cpu().detach())
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/G_%s_%d_aux.png' % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        states = None
        if cfg.CKPT != '':
            print('Load CKPT from: ', cfg.CKPT)
            states = torch.load(cfg.CKPT, map_location=lambda storage, loc: storage)
        text_encoder, image_encoder, netG, netsD, shpsD, segD, start_epoch = self.build_models(states)
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD, optimizersShpD, optimizerSegD = self.define_optimizers(netG, netsD, shpsD, segD, states)

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM

        origin_part_labels = self.part_labels[self.sorted_part_label_indices]
        assert origin_part_labels.shape[1]==1
        plabels_emb = self.dataset.glove_embed(origin_part_labels.view(-1).cpu()).cuda()
        plabels_emb = plabels_emb.detach()
        gen_iterations = 0

        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            # train
            netG.train()
            print('training')
            for step, data in enumerate(self.data_loader):
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                imgs, captions, glove_captions, cap_lens, sem_segs, pooled_sem_segs, class_ids, keys = prepare_train_data(data)
                if not check_nopart(sem_segs): continue
                max_len = int(torch.max(cap_lens))
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, max_len)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                glove_max_len = max(cap_lens)
                glove_captions = glove_captions[:,:glove_max_len].cpu()
                uni_glove_words_embs = self.dataset.glove_embed(glove_captions).cuda().permute((0,2,1)).detach()
                aux_words_embs = form_aux_words_embs(plabels_emb, sem_segs)
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise = Variable(torch.FloatTensor(batch_size, nz)).cuda()
                noise.data.normal_(0, 1)
                raw_fake_imgs, fake_img_front, _, _, _, mu, logvar = netG(noise, sent_emb, words_embs, mask, sem_segs, pooled_sem_segs, uni_glove_words_embs, aux_words_embs, imgs)
                back_index = (pooled_sem_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
                front_index = (pooled_sem_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
                fake_img_background = torch.zeros(fake_img_front.shape).cuda()
                fake_img_background[back_index] = imgs[-1][back_index]
                fake_img_background[front_index] = fake_img_front[front_index]  # [batch_size, channel, height, width]
                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                errD_total_shp = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], raw_fake_imgs[i], sent_emb)
                    if i == (len(netsD) -1):
                        errD += discriminator_loss(netsD[i], imgs[i], fake_img_background, sent_emb)
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                    shpsD[i].zero_grad()
                    errD_shp = discriminator_shape_loss(shpsD[i], imgs[i], raw_fake_imgs[i], sem_segs[i])
                    errD_shp.backward()
                    optimizersShpD[i].step()
                    errD_total_shp += errD_shp
                    D_logs += 'errD_shp%d: %.2f ' % (i, errD_shp.item())

                segD.zero_grad()
                errSegD = segD_glpuzzle_loss(segD, fake_img_background, imgs[-1], pooled_sem_segs[-1], 'test')

                errSegD.backward()
                optimizerSegD.step()
                D_logs += 'errSegD: %.2f ' % errSegD.item()
                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                gen_iterations += 1
                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                match_labels = self.prepare_labels(True)
                errG_total, G_logs = generator_loss(netsD, shpsD, segD, image_encoder, raw_fake_imgs, fake_img_background, sem_segs, pooled_sem_segs, words_embs, sent_emb, match_labels, cap_lens, class_ids, imgs, epoch, self.image_dir, keys)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0: #100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % cfg.save_iter == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(imgs, netG, noise, sent_emb,
                        words_embs, mask, captions,
                        gen_iterations//cfg.save_iter, sem_segs, pooled_sem_segs, uni_glove_words_embs,
                        aux_words_embs, name='average')
                    load_params(netG, backup_para)

            end_t = time.time()
            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_D_seg: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errD_total_shp.item(), errG_total.item(),
                     end_t - start_t))

            #test
            netG.eval()
            print('testing')
            predictions, fake_acts_set, acts_set, w_accuracy, s_accuracy = [], [], [], [], []
            for step, data in enumerate(self.test_data_loader, 0):
                acts, captions, glove_captions, cap_lens, sem_segs, pooled_sem_segs, class_ids, keys, imgs = prepare_test_data(data)
                max_len = int(torch.max(cap_lens))
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, max_len)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                glove_max_len = max(cap_lens)
                glove_captions = glove_captions[:, :glove_max_len].cpu()
                uni_glove_words_embs = self.dataset.glove_embed(glove_captions).cuda().permute((0, 2, 1)).detach()
                aux_words_embs = form_aux_words_embs(plabels_emb, sem_segs)
                # (2) Generate fake images
                ######################################################
                noise = Variable(torch.FloatTensor(self.test_batch_size, nz)).cuda()
                noise.data.normal_(0, 1)
                _, fake_img_front, _, _, _, mu, logvar = netG(noise, sent_emb, words_embs, mask, sem_segs, pooled_sem_segs, uni_glove_words_embs, aux_words_embs, imgs)
                back_index = (pooled_sem_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
                front_index = (pooled_sem_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
                fake_img_background = torch.zeros(fake_img_front.shape).cuda()
                fake_img_background[back_index] = imgs[-1][back_index]
                fake_img_background[front_index] = fake_img_front[front_index]  # [batch_size, channel, height, width]
                # (3) Prepare intermediate results for evaluation
                images = fake_img_background
                region_features, cnn_code = image_encoder(images)
                region_features, cnn_code = region_features.detach(), cnn_code.detach()
                match_labels = self.prepare_labels(False)
                _, _, _, w_accu = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, self.test_batch_size, keys)
                _, _, s_accu = sent_loss(cnn_code, sent_emb, match_labels, class_ids, self.test_batch_size, keys)
                w_accuracy.append(w_accu)
                s_accuracy.append(s_accu)
                pred = self.inception_model(images)
                pred = pred.data.cpu().numpy()
                predictions.append(pred)
                fake_acts = get_activations(images, self.inception_model_fid, self.test_batch_size)
                acts_set.append(acts)
                fake_acts_set.append(fake_acts)
                # (4) Evaluation
            predictions = np.concatenate(predictions, 0)
            mean, std = compute_inception_score(predictions)
            mean_conf, std_conf = negative_log_posterior_probability(predictions)
            accu_w, std_w, accu_s, std_s = np.mean(w_accuracy), np.std(w_accuracy), np.mean(s_accuracy), np.std(s_accuracy)
            acts_set = np.concatenate(acts_set, 0)
            fake_acts_set = np.concatenate(fake_acts_set, 0)
            real_mu, real_sigma = calculate_activation_statistics(acts_set)
            fake_mu, fake_sigma = calculate_activation_statistics(fake_acts_set)
            fid_score = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
            ckpt_ind = None
            if fid_score <= self.min_fid:
                self.min_fid = fid_score
                self.min_epoch = epoch
                ckpt_ind = self.register_ckpt(epoch, fid_score)
            print('inception_score: mean, std, mean_conf, std_conf, accu_w, std_w, accu_s, std_s, fid_score, min_fid, min_epoch')
            print('inception_score: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d' % (mean, std, mean_conf, std_conf, accu_w, std_w, accu_s, std_s, fid_score, self.min_fid, self.min_epoch))
            if ckpt_ind is not None:
                self.save_model(netG, avg_param_G, netsD, shpsD, segD, optimizerG, optimizersShpD, optimizersD, optimizerSegD, epoch, fid_score, ckpt_ind)
                self.save_img_results(imgs, netG, noise, sent_emb,
                                      words_embs, mask, captions,
                                      epoch, sem_segs, pooled_sem_segs, uni_glove_words_embs,
                                      aux_words_embs, name='average')

            if cfg.TRAIN.USE_MLT and ((epoch+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or (epoch+1) == cfg.TRAIN.MAX_EPOCH):
                import mltracker
                snapshot_index = (epoch+1) // cfg.TRAIN.SNAPSHOT_INTERVAL
                mlt_vname = '{0}: {1:02d}'.format(cfg.CONFIG_NAME, snapshot_index)
                with mltracker.start_run():
                    mltracker.set_version(mlt_vname)
                    # mltracker.log_param("param", 5) # log parameters to be tuned
                    for ckpt_obj_ in self.ckpt_lst:
                        ckpt_name = ckpt_obj_.get_name()
                        ckpt_path = os.path.join(self.model_dir, ckpt_name)
                        mltracker.log_file(ckpt_path)
                        ckpt_epoch = ckpt_obj_.get_epoch()
                        mltracker.log_file(os.path.join(self.image_dir, 'G_average_{}.png'.format(ckpt_epoch)))
                        mltracker.log_file(os.path.join(self.image_dir, 'G_average_{}_aux.png'.format(ckpt_epoch)))
                        for pn in cfg.GAN.PUZZLE_NUM:
                            visual_pth = os.path.join(self.image_dir, '{}_{}_visual.png'.format(ckpt_epoch, pn))
                            if os.path.exists(visual_pth):
                                mltracker.log_file(visual_pth)
                    self.ckpt_lst = []

        self.save_model(netG, avg_param_G, netsD, shpsD, segD, optimizerG, optimizersShpD, optimizersD, optimizerSegD, epoch, fid_score, ckpt_ind)




