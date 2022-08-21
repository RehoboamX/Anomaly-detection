import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


def load_dataset(dataroot, class_num, imageSize, trans, train=True):

    dataset = MetalDisk(dataroot, class_num=class_num, is_train=train, resize=imageSize)
    return dataset


class MetalDisk(Dataset):
    def __init__(self, dataset_path='../numclass9', class_num='1', is_train=True, resize=512):
        self.dataset_path = dataset_path
        self.class_num = class_num
        self.is_train = is_train
        self.resize = resize

        # load dataset
        self.x = self.load_dataset_folder()



    def __getitem__(self, idx):
        x = self.x[idx]
        # set transforms
        self.transform_x = transforms.Compose([transforms.Resize((self.resize, self.resize), Image.ANTIALIAS),
                                      transforms.ToTensor(), ])
                                      # T.Normalize(mean=(0.5, 0.5, 0.5),
                                      #             std=(0.5, 0.5, 0.5))])

        x = Image.open(x)
        x = self.transform_x(x)

        return x


    def __len__(self):
        return len(self.x)


    def load_dataset_folder(self):
        phase = 'CutImages' if self.is_train else 'Train'
        x = []

        img_dir = os.path.join(self.dataset_path, phase)

        if not self.is_train:
            img_dir = os.path.join(img_dir, 'BMPImages', self.class_num)

        # load images
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                for f in os.listdir(img_dir)
                                if (f.endswith('.bmp') or f.endswith('.jpg'))])
        x.extend(img_fpath_list)

        return list(x)
