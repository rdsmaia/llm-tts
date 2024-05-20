#import os
#import time
#import logging
import argparse
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from utils.train import train

#torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    # read/load config
    hp = OmegaConf.load(args.config)

    # set cuda and multiprocessing
    args.num_gpus = 0
    torch.manual_seed(hp.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.train.seed)
        args.num_gpus = torch.cuda.device_count()
        print('Batch size per GPU :', hp.train.batch_size)
    else:
        pass

    # launch training
    if args.num_gpus > 1:
        mp.spawn(train,
                 nprocs=args.num_gpus,
                 args=(args, args.checkpoint_path, hp))
    else:
        train(0, args, args.checkpoint_path, hp)