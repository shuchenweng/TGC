from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import form_aux_words_embs
from miscc.utils import calculate_activation_statistics, calculate_frechet_distance
from miscc.utils import get_activations, compute_inception_score, negative_log_posterior_probability
from miscc.losses import segD_glpuzzle_loss
from Datasets.datasetTest import prepare_test_data
from models.Dmodel import SEG_D_NET
from models.Emodel import RNN_ENCODER, CNN_ENCODER
from models.Pmodel import INCEPTION_V3, INCEPTION_V3_FID
from models.Rmodel import G_NET

import os
from PIL import Image
from miscc.losses import words_loss, sent_loss
import numpy as np

# ################# Text to image task############################ #
class condGANTester(object):
    def __init__(self, data_loader, dataset, args):
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.dataset = dataset
        self.data_dir = dataset.data_dir
        self.n_words = dataset.n_words
        self.ixtoword = dataset.ixtoword
        self.part_labels = dataset.part_labels
        self.part_label_lens = dataset.part_label_lens
        self.sorted_part_label_indices = dataset.sorted_part_label_indices
        self.data_loader = data_loader
        self.args = args
        self.inception_model = INCEPTION_V3()
        self.inception_model.cuda()
        self.inception_model = nn.DataParallel(self.inception_model)
        self.inception_model.eval()
        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[cfg.TEST.FID_DIMS]
        self.inception_model_fid = INCEPTION_V3_FID([block_idx])
        self.inception_model_fid.cuda()
        self.inception_model_fid = nn.DataParallel(self.inception_model_fid)
        self.inception_model_fid.eval()
        dataset.create_acts()


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
        text_encoder = nn.DataParallel(text_encoder, device_ids=cfg.GPU_group)

        netG = nn.DataParallel(G_NET(),device_ids=cfg.GPU_group)
        netG.apply(weights_init)
        epoch = 0
        if states is not None:
            epoch = int(states['epoch'])+1
            netG.load_state_dict(states['netG'])
        netG.cuda()

        segD = nn.DataParallel(SEG_D_NET(), device_ids=cfg.GPU_group).cuda()
        segD.load_state_dict(states['netSegD'])
        return [text_encoder, image_encoder, netG, segD, epoch]

    def prepare_labels(self):
        batch_size = self.batch_size
        match_labels = Variable(torch.LongTensor(range(batch_size))).cuda()
        return match_labels

    def test(self):
        if cfg.CKPT == '':
            print('Error: the path for morels is not found!')
            return
        states = torch.load(cfg.CKPT, map_location=lambda storage, loc: storage)
        text_encoder, image_encoder, netG, segD, start_epoch = self.build_models(states)

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True).cuda()
        # the path to save generated images
        model_dir = cfg.CKPT
        s_tmp = model_dir[:model_dir.rfind('.pth')]
        save_dir = '%s/%s' % (s_tmp, 'valid')
        mkdir_p(save_dir)
        origin_part_labels = self.part_labels[self.sorted_part_label_indices]
        assert origin_part_labels.shape[1] == 1
        plabels_emb = self.dataset.glove_embed(origin_part_labels.view(-1).cpu()).cuda()
        plabels_emb = plabels_emb.detach()
        predictions,fake_acts_set, acts_set, w_accuracy, s_accuracy = [], [], [], [], []

        for step, data in enumerate(self.data_loader, 0):
            if step % 100 == 0:
                print('step: ', step, len(self.data_loader))

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
            glove_captions = glove_captions[:,:glove_max_len].cpu()
            uni_glove_words_embs = self.dataset.glove_embed(glove_captions).cuda().permute((0,2,1)).detach()
            aux_words_embs = form_aux_words_embs(plabels_emb, sem_segs)
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            raw_fake_imgs, fake_img_front, _, _, _, _, _ = netG(noise, sent_emb, words_embs, mask, sem_segs, pooled_sem_segs, uni_glove_words_embs, aux_words_embs, imgs)
            back_index = (pooled_sem_segs[-1] == 0).unsqueeze(dim=1).expand_as(imgs[-1])
            front_index = (pooled_sem_segs[-1] == 1).unsqueeze(dim=1).expand_as(imgs[-1])
            fake_img_background = torch.zeros(fake_img_front.shape).cuda()
            fake_img_background[back_index] = imgs[-1][back_index]
            fake_img_background[front_index] = fake_img_front[front_index]  # [batch_size, channel, height, width]

            for i in range(batch_size):
                save_dir_tmp = os.path.join(save_dir, keys[i].split('/')[0])
                if not os.path.exists(save_dir_tmp):
                    os.mkdir(save_dir_tmp)
                save_pth = os.path.join(save_dir, keys[i]+'.png')
                print(save_pth)
                Image.fromarray(((fake_img_background[i] + 1)*127.5).permute((1,2,0)).detach().cpu().numpy().astype('uint8')).save(save_pth)

            # calculate score
            # (3) Prepare intermediate results for evaluation
            images = fake_img_background
            region_features, cnn_code = image_encoder(images)
            region_features, cnn_code = region_features.detach(), cnn_code.detach()
            match_labels = self.prepare_labels()
            _, _, _, w_accu = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size, keys)
            _, _, s_accu = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size, keys)
            w_accuracy.append(w_accu)
            s_accuracy.append(s_accu)
            pred = self.inception_model(images)
            pred = pred.data.cpu().numpy()
            predictions.append(pred)
            fake_acts = get_activations(images, self.inception_model_fid, batch_size)
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

        print('inception_score: mean, std, mean_conf, std_conf, accu_w, std_w, accu_s, std_s, fid_score')
        print('inception_score: %f, %f, %f, %f, %f, %f, %f, %f, %f' %(mean, std,
            mean_conf, std_conf, accu_w, std_w, accu_s, std_s, fid_score))





