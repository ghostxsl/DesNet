import torch.utils.data as data
import torch

import numpy as np
from PIL import Image
import os
import os.path


def default_loader(path):
    return Image.open(path)

class SarImage(data.Dataset):
    def __init__(self, txt_dir, transform=None, target_transform=None, loader=default_loader):
        ftxt = open(txt_dir)
        imgs = []
        for line in ftxt.readlines():
            cls = line.split()
            fdata = cls.pop(0)
            flabel = cls.pop(0)
            imgs.append((fdata, flabel))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]

        img_data = self.loader(path)
        img_label = self.loader(target)
        img1 = torch.from_numpy(np.array(img_data, np.float32, copy=False))
        img2 = torch.from_numpy(np.array(img_label, np.float32, copy=False))

        Pattern = img1 / (img2 + 1e-9) - 1
        img3 = img1 - img1.mean()
        Img = img3 / (img3.max() + 1e-9)

        Img = Img.view(img_data.size[1], img_data.size[0], 1)
        Img = Img.transpose(0, 1).transpose(0, 2).contiguous()
        Pattern = Pattern.view(img_label.size[1], img_label.size[0], 1)
        Pattern = Pattern.transpose(0, 1).transpose(0, 2).contiguous()

        return Img, Pattern

    def __len__(self):
        return len(self.imgs)
