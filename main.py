from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from Datasets.datasetTrain import TextDatasetTrain
from Datasets.datasetTest import TextDatasetTest
from trainer import condGANTrainer as trainer
from tester import condGANTester as tester

import os
import sys
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
# sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/test_bird_SC.yml', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--batch', type=int, default=-1, help='batch_size')
    parser.add_argument('--gpu', type=str, default='0', help='gpu list')
    parser.add_argument('--ckpt', type=str, default='none')
    parser.add_argument('--cn', type=str, default='glpuzzle', help='glpuzzle|pixel|none')
    parser.add_argument('--pnn', type=str, default='4', help='4|8|16, height//pnn')
    parser.add_argument('--tfrt', type=int, default=7, help='7|15|31')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg_from_file(args.cfg_file)
    if args.data_dir != '': cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if args.ckpt != 'none': cfg.CKPT = args.ckpt
    cfg.MODEL.COMP_NAME = args.cn
    cfg.MODEL.TFRT = args.tfrt
    cfg.GAN.PUZZLE_NUM = [int(pnn_id) for pnn_id in args.pnn.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg.GPU_group = [int(gpu_id) for gpu_id in range(len(args.gpu.split(',')))]
    if args.batch!=-1: cfg.TRAIN.BATCH_SIZE = args.batch
    if args.batch!=-1: cfg.TEST.BATCH_SIZE = args.batch

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/%s_%s/%s' % (cfg.MODEL_DIR, cfg.OUT_PREFIX, cfg.CONFIG_NAME, timestamp)

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    # test
    import miscc.compose as transforms
    test_image_transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize)])
    test_dataset = TextDatasetTest(cfg.DATA_DIR, split='test',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=test_image_transform)
    assert test_dataset
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.TEST.BATCH_SIZE,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))
    # train
    if cfg.TRAIN.FLAG:
        import miscc.compose as transforms
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        train_dataset = TextDatasetTrain(cfg.DATA_DIR, split='train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        assert train_dataset
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
        algo = trainer(output_dir, train_dataloader, train_dataset, test_dataloader, test_dataset, args)
        algo.train()
    algo = tester(test_dataloader, test_dataset, args)
    algo.test()
