import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('-load_model', type=str, default='./models/rgb_imagenet.pt')
parser.add_argument('-root', type=str, default='/home2/Dataset/FineAction/resize_video_224_crop/')
parser.add_argument('-split', type=str, default='./fineaction/annotations_gt.json')
parser.add_argument('-gpu', type=str, default='2')
parser.add_argument('-save_dir', type=str, default='/home2/Dataset/FineAction/i3d_feature_224_crop/')
parser.add_argument('-start', type=int, default=0)
parser.add_argument('-length', type=int, default=1500)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import time
import datetime
from joblib import Parallel, delayed
import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset
import tqdm
import ffmpeg

def run(max_steps=64e3, mode='rgb', root='/data/Dataset/Charades_videos/Charades_v1_rgb', split='charades/charades.json', batch_size=1, load_model='', save_dir=''):

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    # set the configuration
    read_path = "/home2/Dataset/FineAction/resize_video_224_crop/"
    save_path = "/home2/Dataset/FineAction/i3d_feature_224_crop/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    video_name = os.listdir(read_path)
    video_name.sort()

    begin = args.start
    end = begin + args.length
    for name in tqdm.tqdm(video_name):
        if int(name.split('.')[0][-6:]) < begin or int(name.split('.')[0][-6:]) >= end:
            continue
        if os.path.exists(os.path.join(save_path, name.split('.')[0]+'.npy')):
            continue
        # read the video
        out, _ = (
            ffmpeg
            .input(os.path.join(read_path, name))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel='error')
            .run(capture_stdout=True)
        )
        video = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, 224, 224, 3])
        )
        # extract the i3d features and save to .npy file
        inputs = video.astype(np.float32).transpose([3,0,1,2])   # (T x H x W x C) -> (C x T x H x W)
        if inputs.shape[1] > 100000:
            for t in range(inputs.shape[1]):
                inputs[:, t, :, :] = (inputs[:, t, :, :] / 255.0) * 2.0 - 1.0
        else:
            inputs = (inputs / 255.0) * 2.0 - 1.0
        features = []
        T = inputs.shape[1]
        # def extract_i3d(start):
        #     segment = inputs[:, start:start+8, :, :]
        #     segment = torch.from_numpy(segment)
        #     segment = Variable(segment.unsqueeze(0).cuda(), volatile=True)
        #     features.append((start, i3d.extract_features(segment).squeeze(0).permute(1,2,3,0).data.cpu().numpy()))
        # parallel = Parallel(n_jobs=5, prefer="processes")
        # parallel(delayed(extract_i3d)(start) for start in range(0, T-8, 4))
        # features.sort(key = lambda e: e[0])
        # features = [feature[1] for feature in features]
        # features = np.concatenate(features, axis=0)
        for start in range(0, T-8, 4):
            segment = inputs[:, start:start+8, :, :]
            segment = torch.from_numpy(segment)
            segment = Variable(segment.unsqueeze(0).cuda(), volatile=True)
            features.append(i3d.extract_features(segment).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
        features = np.concatenate(features, axis=0)
        print('{} shape: {}'.format(name, features.shape))
        np.save(os.path.join(save_path, name.split('.')[0]), features)


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, split=args.split, load_model=args.load_model, save_dir=args.save_dir)
