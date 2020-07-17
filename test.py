''' Baisc packages
'''
import os
import glob
import tqdm
import copy
import random
import importlib
import numpy as np
from decimal import Decimal
from collections import OrderedDict

''' Configuration packages
'''
import yaml
import argparse
from easydict import EasyDict as edict

''' PyTorch packages
'''
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

''' Customized packages
'''
from utility import optimizer_util as opti_util
from Data_zoo import common_utils as cutils

def net_test(args):
    ''' 0. Import model from Model_zoo
	'''
    AMAF = importlib.import_module('.{}'.format(args.model.name.lower()), package=args.model.zoo)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    ''' 1. Model
    '''
    print('========> Build model')
    
    SRmodel = AMAF.create_model(args.training)
    pretrained_model = args.testing.pretrained_model_path
    SRmodel.model.load_state_dict(torch.load(pretrained_model))
    SRmodel.eval()
    for k,v in SRmodel.model.named_parameters():
        v.requires_grad = False
    SRmodel = SRmodel.to(device)

#    model_path = os.path.join('MSRResNetx4_model', 'MSRResNetx4.pth')
#    SRmodel = MSRResNet(in_nc=3, out_nc=3, nf=64, nb=16, upscale=4)
#    SRmodel.load_state_dict(torch.load(model_path), strict=True)
#    SRmodel.eval()
#    for k, v in SRmodel.named_parameters():
#        v.requires_grad = False
#    SRmodel = SRmodel.to(device)
    number_parameters = sum(map(lambda x: x.numel(), SRmodel.parameters()))
    print('========> Parameters: {}'.format(number_parameters))
    ''' 2.Data
    '''
    dir_lr1 = os.path.join(args.dataset.dir_root, args.dataset.dir_lr)
    if args.vtype=='val':
        dir_lr = dir_lr1
        dir_lr = dir_lr.replace('train', 'valid')
        dir_lr_list = cutils.get_image_paths(dir_lr)
    elif args.vtype=='test':
        dir_lr = dir_lr1
        dir_lr = dir_lr.replace('train', 'test')
        dir_lr_list = cutils.get_image_paths(dir_lr)
    else:
        ValueError('Error args.vtype, only <test> are useful.')

    dir_sr = args.testing.result_dir
    if not os.path.exists(dir_sr):
        os.mkdir(dir_sr)

    ''' 3.Test
    '''
    idx = 0
    test_results = OrderedDict()
    test_results['runtime'] = []


    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    for img in dir_lr_list:
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        print('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        lr = cutils.imread_uint(img, n_channels=3)
        lr = cutils.uint2tensor4(lr)
        lr = lr.float()
        lr = lr.to(device)

        t_start.record()
        sr = SRmodel(lr)
        t_end.record()
        torch.cuda.synchronize()
        sr = cutils.quantize(sr, args.training.rgb_range)

        cutils.imsave(sr, os.path.join(args.testing.result_dir, img_name+ext))
        test_results['runtime'].append(t_start.elapsed_time(t_end))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    print(('------> Average runtime is : {:.6f} seconds'.format(ave_runtime)))


if __name__ == "__main__":
    ''' Parameters
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./yaml/test.yaml')
    cfg = parser.parse_args()
    args = edict(yaml.load(open(cfg.config, 'r')))
    print(args.DESCRIPTION)
    cudnn.benchmark = args.cudnn_benchmark

    ''' Testing
    '''
    print('HR_dir -> {}'.format(os.path.join(args.dataset.dir_root, args.dataset.dir_hr)))
    print('LR_dir -> {}'.format(os.path.join(args.dataset.dir_root, args.dataset.dir_lr)))
    print('Load_ckp -> {}'.format(args.testing.pretrained_model_path))

    print('--------------------------Begin-----------------------')
    net_test(args)