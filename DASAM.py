import os
from PIL import Image
import argparse
from torch.autograd import Variable
from Datasets.datasetTrain import TextDatasetTrain, prepare_train_data
from Datasets.datasetTest import TextDatasetTest, prepare_test_data
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from models.Emodel import CNN_ENCODER, RNN_ENCODER
from miscc.losses import words_loss, sent_loss
from miscc.config import cfg, cfg_from_file
from miscc.utils import build_super_images

def train(dataloader, image_encoder, text_encoder, optimizer, dataset, gen_iterations):
    text_encoder.train()
    image_encoder.train()
    epoch_w_loss = 0
    epoch_s_loss = 0
    cnt = 0
    match_labels = Variable(torch.LongTensor(range(cfg.TRAIN.BATCH_SIZE))).cuda()
    for i, data in enumerate(dataloader):
        text_encoder.zero_grad()
        image_encoder.zero_grad()
        imgs, captions, glove_captions, cap_lens, sem_segs, pooled_sem_segs, class_ids, keys = prepare_train_data(data)
        max_len = int(torch.max(cap_lens))
        words_embs, sent_emb = text_encoder(captions, cap_lens, max_len)
        region_features, cnn_code = image_encoder(imgs[-1])

        batch_size = imgs[-1].shape[0]
        w_loss0, w_loss1, attn_maps, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size, keys)
        w_loss = w_loss0 + w_loss1
        s_loss0, s_loss1, _ = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size, keys)
        s_loss = s_loss0 + s_loss1
        loss = w_loss + s_loss
        epoch_w_loss = epoch_w_loss + w_loss
        epoch_s_loss = epoch_s_loss + s_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm(text_encoder.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()
        # print('mini train ', cnt, w_loss.item(), s_loss.item(), loss.item())
        cnt += 1
        if i % cfg.save_iter == 0:
            attn = attn_maps[-1]
            att_sze = attn.size(2)
            img_set, _ = build_super_images(imgs[-1].cpu().detach(), captions, dataset.ixtoword, attn_maps, att_sze, max_word_num=cfg.TEXT.WORDS_NUM)
            if img_set is not None:
                im = Image.fromarray(img_set)
                save_dir = os.path.join(cfg.PRETRAINED_DIR, 'pre_visual')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_pth = os.path.join(save_dir, str(gen_iterations) + '_' + str(i) + '.jpg')
                im.save(save_pth)
    return epoch_w_loss.item() / cnt, epoch_s_loss.item() / cnt

def evaluate(dataloader, image_encoder, text_encoder):
    with torch.no_grad():
        epoch_w_loss = 0
        epoch_s_loss = 0
        cnt = 0
        match_labels = Variable(torch.LongTensor(range(cfg.TEST.BATCH_SIZE))).cuda()
        for data in dataloader:
            acts, captions, glove_captions, cap_lens, sem_segs, pooled_sem_segs, class_ids, keys, imgs = prepare_test_data(data)

            max_len = int(torch.max(cap_lens))
            region_features, cnn_code = image_encoder(imgs[-1])
            words_embs, sent_emb = text_encoder(captions, cap_lens, max_len)

            batch_size = imgs[-1].shape[0]
            w_loss0, w_loss1, _, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size, keys)
            w_loss = w_loss0 + w_loss1
            s_loss0, s_loss1, _ = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size, keys)
            s_loss = s_loss0 + s_loss1

            epoch_w_loss = epoch_w_loss + w_loss
            epoch_s_loss = epoch_s_loss + s_loss

            # print('mini test ', cnt,  w_loss.item(), s_loss.item(), loss.item())
            cnt += 1
    return epoch_w_loss.item() / cnt, epoch_s_loss.item() / cnt

def read_weight(image_encoder, text_encoder):
    if cfg.CKPT == -1: return
    img_weight_pth = os.path.join(cfg.PRETRAINED_DIR, 'pre_model_weight', 'image_' + str(cfg.CKPT))
    text_weight_pth = os.path.join(cfg.PRETRAINED_DIR, 'pre_model_weight', 'text_' + str(cfg.CKPT))
    img_state_dict = torch.load(img_weight_pth)
    image_encoder.load_state_dict(img_state_dict)
    text_state_dict = torch.load(text_weight_pth)
    text_encoder.load_state_dict(text_state_dict)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/train_bird_SC.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0,1', help='gpu list')
    parser.add_argument('--ckpt', type=int, default=-1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg_from_file(args.cfg_file)
    cfg.CKPT = args.ckpt
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg.GPU_group = [int(gpu_id) for gpu_id in range(len(args.gpu.split(',')))]

    imsize_width = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    imsize_height = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    import miscc.compose as transforms
    train_image_transform = transforms.Compose([
        transforms.Resize((int(imsize_height * 76 / 64), int(imsize_width * 76 / 64))),
        transforms.RandomCrop((imsize_height, imsize_width)),
        transforms.RandomHorizontalFlip()])
    train_dataset = TextDatasetTrain(cfg.DATA_DIR, split='train',
                                     base_size=cfg.TREE.BASE_SIZE,
                                     transform=train_image_transform)
    assert train_dataset
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    test_image_transform = transforms.Compose([
        transforms.Resize((imsize_height, imsize_width)),
        transforms.CenterCrop((imsize_height, imsize_width))])
    test_dataset = TextDatasetTest(cfg.DATA_DIR, split='test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=test_image_transform)
    assert test_dataset
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.TEST.BATCH_SIZE,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))
    test_dataset.create_acts()

    text_encoder = RNN_ENCODER(train_dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM).cuda()
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM).cuda()

    read_weight(image_encoder, text_encoder)

    text_encoder = nn.DataParallel(text_encoder, device_ids=cfg.GPU_group)
    para = list(text_encoder.parameters())
    image_encoder = nn.DataParallel(image_encoder, device_ids=cfg.GPU_group)


    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    lr = cfg.TRAIN.ENCODER_LR
    epoch_iterations = 0
    while True:
        optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
        w_loss, s_loss = train(train_dataloader, image_encoder, text_encoder, optimizer, train_dataset, epoch_iterations)
        epoch_loss = w_loss + s_loss
        logs = '%d w_loss: %.2f s_loss: %.2f total_loss: %.2f' % (epoch_iterations, w_loss, s_loss, epoch_loss)
        print(logs)

        test_w_loss, test_s_loss = evaluate(test_dataloader, image_encoder, text_encoder)
        epoch_loss = test_w_loss + test_s_loss
        test_logs = '%d w_loss: %.2f s_loss: %.2f total_loss: %.2f' % (epoch_iterations, test_w_loss, test_s_loss, epoch_loss)
        print(test_logs)

        if lr > cfg.TRAIN.ENCODER_LR / 10.:
            lr *= 0.98
        if epoch_iterations % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
            model_weight_dir = os.path.join(cfg.PRETRAINED_DIR, 'pre_model_weight')
            image_weight_save_path = os.path.join(model_weight_dir, 'image_' + str(epoch_iterations))
            label_weight_save_path = os.path.join(model_weight_dir, 'text_' + str(epoch_iterations))
            if not os.path.exists(model_weight_dir):
                os.makedirs(model_weight_dir)
            torch.save(image_encoder.module.state_dict(), image_weight_save_path)
            torch.save(text_encoder.module.state_dict(), label_weight_save_path)
            print('Save models weight.')
            if cfg.TRAIN.USE_MLT:
                import mltracker
                mlt_vname = '{0}: {1:02d}'.format(cfg.CONFIG_NAME, epoch_iterations)
                with mltracker.start_run():
                    mltracker.set_version(mlt_vname)
                    mltracker.log_file(image_weight_save_path)
                    mltracker.log_file(label_weight_save_path)


        epoch_iterations = epoch_iterations + 1
