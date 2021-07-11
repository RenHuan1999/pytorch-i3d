import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
# import h5py

import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    if os.path.exists(os.path.join(image_dir, vid+'.mp4')):
        cap = cv2.VideoCapture(os.path.join(image_dir, vid+'.mp4'))
    else:
        cap = cv2.VideoCapture(os.path.join(image_dir, vid+'.webm'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    print('frame_num of {} is {}'.format(vid, num))
    for i in range(start, start+num):
        ret, img = cap.read()   # ret 读取了数据就返回 True，没有读取数据(已到尾部)就返回 False
        # img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
        if not ret:
            break
        img = img[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    cap.release()  
    return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start+num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
        w,h = imgx.shape
        if w < 224 or h < 224:
            d = 224.-min(w,h)
            sc = 1+d/min(w,h)
            imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
            imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
            
        imgx = (imgx/255.)*2 - 1
        imgy = (imgy/255.)*2 - 1
        img = np.asarray([imgx, imgy]).transpose([1,2,0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=106):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)
    data = data['database']

    action_map = {}
    actions = open('./fineaction/action_mapping.txt', 'r').readlines()
    for action in actions:
        name = action.strip().split(',')[1]
        label = int(action.strip().split(',')[0])
        action_map[name] = label

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, data[vid]['filename'])):
            continue
        num_frames = data[vid]['actual_frame_num']
        if mode == 'flow':
            num_frames = num_frames//2
            
        label = np.zeros((num_classes,num_frames), np.float32)

        fps = data[vid]['fps']
        for ann in data[vid]['annotations']:
            start = int(ann['segment'][0] * fps) + 1
            end = int(ann['segment'][1] * fps) + 1
            label[action_map[ann['label'].strip()], start:end] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1
    
    dataset.sort(key = lambda e: e[0])
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index, start=None, num=None):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            return 0, 0, vid

        if start==None or num==None:
            start, num = 1, nf

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start, num)
        else:
            imgs = load_flow_frames(self.root, vid, start, num)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
