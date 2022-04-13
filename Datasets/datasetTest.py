from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from miscc.config import cfg
from miscc.load import load_acts_data, load_class_id, load_filenames, load_glove_emb, load_bbox, load_part_label, load_text_data, load_bytes_data
from miscc.utils import match_cap, get_activations
from models.Pmodel import INCEPTION_V3_FID

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import io
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_test_data(data):
    acts, captions, glove_captions, captions_lens, sem_segs, pooled_sem_segs, class_ids, keys, imgs = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        real_imgs.append(Variable(imgs[i]).cuda())

    real_sem_segs, real_pooled_sem_segs = [], []
    for i in range(len(sem_segs)):
        sem_segs[i] = sem_segs[i][sorted_cap_indices].float()
        pooled_sem_segs[i] = pooled_sem_segs[i][sorted_cap_indices].float()
        real_sem_segs.append(Variable(sem_segs[i]).cuda())
        real_pooled_sem_segs.append(Variable(pooled_sem_segs[i]).cuda())

    acts = acts[sorted_cap_indices].numpy()
    captions = captions[sorted_cap_indices].squeeze()
    glove_captions = glove_captions[sorted_cap_indices].squeeze(0)
    if isinstance(class_ids, torch.Tensor):
        class_ids = class_ids[sorted_cap_indices].numpy()
    else:
        reshape_cls_list = []
        for i in range(sorted_cap_indices.shape[0]):
            tmp_cls_list = []
            for j in range(cfg.GAN.MAX_LENGTH_CLS):
                tmp_cls_list.append(class_ids[j][sorted_cap_indices[i]])
            reshape_cls_list.append(tmp_cls_list)
        class_ids = reshape_cls_list
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    captions = Variable(captions).cuda()
    glove_captions = Variable(glove_captions).cuda()
    sorted_cap_lens = Variable(sorted_cap_lens).cuda()

    return [acts, captions, glove_captions, sorted_cap_lens, real_sem_segs,
            real_pooled_sem_segs, class_ids, keys, real_imgs]

def prepare_acts_data(data):
    imgs, keys = data

    real_imgs = []
    for i in range(len(imgs)):
        real_imgs.append(Variable(imgs[i]).cuda())

    return [real_imgs, keys]

class TextDatasetTest(data.Dataset):
    def __init__(self, data_dir, split='test',
                 base_size=64,
                 transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = load_bbox(self.data_dir)
        split_dir = os.path.join(data_dir, split)
        train_names = load_filenames(data_dir, 'train')
        test_names = load_filenames(data_dir, 'test')
        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = load_text_data(data_dir, split, train_names, test_names, self.embeddings_num)
        self.glove_captions, self.glove_ixtoword, self.glove_wordtoix, self.glove_embed = load_glove_emb(data_dir, split, train_names, test_names, self.embeddings_num)
        match_cap(self.captions, self.ixtoword, self.glove_captions, self.glove_ixtoword)
        self.part_labels, self.part_label_lens, self.sorted_part_label_indices = load_part_label(data_dir, self.glove_wordtoix)
        self.class_id = load_class_id(split_dir, len(self.filenames))
        self.acts_dict = None
        self.sem_segs_bytes = load_bytes_data(data_dir, split, self.filenames, 'semsegs', '.npz')
        self.img_bytes = load_bytes_data(data_dir, split, self.filenames, 'images', '.jpg')

    def get_imgs(self, img_byte, sem_segs_byte, imsize, bbox=None, transform=None, normalize=None):
        img = Image.open(io.BytesIO(img_byte)).convert('RGB')
        sem_seg = np.load(io.BytesIO(sem_segs_byte))['seg']
        sem_seg = Image.fromarray(np.uint8(sem_seg))
        width, height = sem_seg.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])
            sem_seg = sem_seg.crop([x1, y1, x2, y2])
        if transform is not None:
            img, sem_seg = transform(img, sem_seg)
        ret = []
        new_sem_segs = []
        pooled_sem_segs = []
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
                re_sem_seg = transforms.Scale(imsize[i], interpolation=Image.NEAREST)(sem_seg)
            else:
                re_img = img
                re_sem_seg = sem_seg
            ret.append(normalize(re_img))
            re_sem_seg = np.asarray(re_sem_seg)
            new_sem_seg = np.zeros((cfg.GAN.P_NUM, imsize[i], imsize[i]))
            for j in range(cfg.GAN.P_NUM):
                new_sem_seg[j, re_sem_seg == j + 1] = 1
            pooled_sem_seg = np.amax(new_sem_seg, axis=0)
            pooled_sem_segs.append(pooled_sem_seg)
            new_sem_segs.append(new_sem_seg)
        return ret, new_sem_segs, pooled_sem_segs

    def get_actions(self, imsize, bbox=None, transform=None):
        action_shape = []
        for action_index in range(cfg.TEST.ACTION_FRAME):
            random_num = random.randint(0, self.__len__() - 1)
            action_imgs, action_segs, action_pooled_segs = self.get_imgs(self.img_bytes[random_num], self.sem_segs_bytes[random_num], imsize, bbox, transform, self.norm)
            action_pair = []
            # action_pair.append(action_imgs) # some code should have written here
            action_pair.append(action_segs)
            action_pair.append(action_pooled_segs)
            action_shape.append(action_pair)
        return action_shape

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        glove_sent_caption = np.asanyarray(self.glove_captions[sent_ix]).astype('int64')
        if ((sent_caption == 0).sum() > 0) or ((glove_sent_caption == 0).sum() > 0):
            print('ERROR: do not need END (0) token', sent_caption)

        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        y = np.zeros((cfg.TEXT.WORDS_NUM), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
            y[:num_words] = glove_sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            y[:] = glove_sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, y, x_len

    def dump_fid_acts(self, data_dir):
        filepath = os.path.join(data_dir, '%s_acts.pickle'% 'test')
        if os.path.isfile(filepath):
            return
        block_idx = INCEPTION_V3_FID.BLOCK_INDEX_BY_DIM[cfg.TEST.FID_DIMS]
        inception_model_fid = INCEPTION_V3_FID([block_idx])
        inception_model_fid.cuda()
        inception_model_fid = nn.DataParallel(inception_model_fid)
        inception_model_fid.eval()
        act_dataset = create_acts_dataset(cfg.DATA_DIR, 'test', cfg.TRAIN.BATCH_SIZE, self.transform)
        act_dataloader = torch.utils.data.DataLoader(
            act_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))
        acts_dict = {}
        count = 0
        for step, data in enumerate(act_dataloader):
            if count % 10 == 0:
                print('%07d / %07d'%(count, self.__len__() / cfg.TRAIN.BATCH_SIZE))
            imgs, keys = prepare_acts_data(data)
            batch_size = len(keys)
            acts = get_activations(imgs[-1], inception_model_fid, batch_size)
            for batch_index in range(batch_size):
                acts_dict[keys[batch_index]] = acts[batch_index]

            count += 1
        with open(filepath, 'wb') as f:
            pickle.dump([acts_dict], f, protocol=2)
            print('Save to: ', filepath)

    def create_acts(self):
        self.dump_fid_acts(cfg.DATA_DIR)
        self.acts_dict = load_acts_data(cfg.DATA_DIR)

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        bbox = self.bbox[key]
        #
        imgs, sem_segs, pooled_sem_segs = self.get_imgs(self.img_bytes[index], self.sem_segs_bytes[index], self.imsize, bbox, self.transform, self.norm)
        # find action frame path
        # action_shape = self.get_actions(self.imsize, None, self.transform)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, glove_caps, cap_len = self.get_caption(new_sent_ix)
        if key in self.acts_dict:
            acts = self.acts_dict[key]
        else:
            acts = np.zeros([5])
        return acts, caps, glove_caps, cap_len, sem_segs, pooled_sem_segs, cls_id, key, imgs

    def __len__(self):
        return len(self.filenames)

class create_acts_dataset(data.Dataset):
    def __init__(self, data_dir, split, base_size, transform):
        split_dir = os.path.join(data_dir, split)
        self.data_dir = data_dir
        self.filenames = load_filenames(self.data_dir, 'test')
        self.class_id = load_class_id(split_dir, len(self.filenames))
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.bbox = load_bbox(self.data_dir)
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        self.img_bytes = load_bytes_data(data_dir, split, self.filenames, 'images', '.jpg')

    def get_imgs(self, img_path, imsize, bbox=None, transform=None, normalize=None):
        img = Image.open(io.BytesIO(img_path)).convert('RGB')
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])
        if transform is not None:
            img, _ = transform(img, img)
        ret = []
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))
        return ret

    def __getitem__(self, index):
        key = self.filenames[index]
        bbox = self.bbox[key]
        imgs = self.get_imgs(self.img_bytes[index], self.imsize, bbox, self.transform, normalize=self.norm)
        return imgs, key

    def __len__(self):
        return len(self.filenames)