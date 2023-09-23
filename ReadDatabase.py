import os
import os.path as osp
import random

import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from cityscapesscripts.helpers.labels import trainId2label

# parameters
city_mean, city_std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(city_mean, city_std)])

palette = [255,255,255, # background
           255,000,000, # desease 1
           000,000,255, # desease 2
           000,255,000] # leaf

for key in sorted(trainId2label.keys()):
    if key != -1 and key != 255:
        palette += list(trainId2label[key].color)
   
# Database Class   
class Database(Dataset):
    def __init__(self, root, MaxLabel, split='train', crop_size=(512, 512),image_size=(1024, 1024), mean=city_mean, std=city_std, ignore_label=0):

        self.split = split
        self.crop_h, self.crop_w = crop_size
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.MaxLabel = MaxLabel
        self.ignore_label = ignore_label
        self.files = []

        for root_dir, dirs, files in os.walk(osp.join(root, 'Images')):
            for file in files:
                img_file = osp.join(root_dir, file)
                label_file = osp.join(root_dir.replace('Images', 'Labels'),
                                      file.replace('.JPG', '_label.png'))
                self.files.append({'img': img_file, 'label': label_file, 'name': file})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles['img'], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles['label'], cv2.IMREAD_GRAYSCALE)
        name = datafiles['name']

        scale = (255/self.MaxLabel)
        label = np.round(label/scale)
        label = label.astype(np.uint8)
            
        # random resize, multiple scale training
        if self.split == 'train':
            f_scale = random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            
        # resize validation
        if self.split == 'val':
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            
        image = np.asarray(image, np.float32)
        
        # change to RGB
        image = image[:, :, ::-1]
        
        # normalization
        image /= 255.0
        image -= self.mean
        image /= self.std

        # random crop
        if self.split == 'train':
            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.8, 0.8, 0.8))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
            else:
                img_pad, label_pad = image, label

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            label = label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]

        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        label = np.asarray(label, np.long)

        # random horizontal flip
        if self.split == 'train':
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), name
