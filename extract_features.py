import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('-load_model', type=str, default='./models/rgb_charades.pt')
parser.add_argument('-root', type=str, default='/data/Dataset/FineAction-video/video_data_end/')
parser.add_argument('-split', type=str, default='./fineaction/annotations_gt.json')
parser.add_argument('-gpu', type=str, default='3')
parser.add_argument('-save_dir', type=str, default='/data/Dataset/FineAction-video/i3d-feature/')

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


def run(max_steps=64e3, mode='rgb', root='/data/Dataset/Charades_videos/Charades_v1_rgb', split='charades/charades.json', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    val_dataset = Dataset(split, 'validation', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in ['train', 'val']:
        i3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0

        begin_time = time.time()
        start_time = time.time()
        # Iterate over data.
        for data in dataloaders[phase]:
        # for i in range(datasets[phase].__len__()):
        #     data = datasets[phase].__getitem__(i)
            # get the inputs
            inputs, labels, name = data
            read_time = time.time()
            # print('Processing ', name[0], ', the shape of input: ', inputs.shape)
            if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
                continue

            b,c,t,h,w = inputs.shape
            if t > 1600:
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
                print('Processing {}, the shape of input: {}, the shape of features: {}'.format(name[0], inputs.shape, np.concatenate(features, axis=0).shape))
            else:
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                print('Processing {}, the shape of input: {}, the shape of features: {}'.format(name[0], inputs.shape, features.squeeze(0).permute(1,2,3,0).shape))
            end_time = time.time()
            print('{}: \tRead time: {}, \tExtract time: {}, \tTotal time: {}'.format(name[0], 
                datetime.timedelta(seconds=int(read_time - start_time)), 
                datetime.timedelta(seconds=int(end_time - read_time)), 
                datetime.timedelta(seconds=int(end_time - begin_time))))
            start_time = time.time()


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, split=args.split, load_model=args.load_model, save_dir=args.save_dir)
