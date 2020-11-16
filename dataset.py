import os
from os import listdir
from os.path import join
import random

import cv2 as cv
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


# data_dir = /home/hanfang_yang/GIS/data/geoimage/train/images
class GeoImage(data.Dataset):
    def __init__(self, data_dir, direction, filename):
        super(GeoImage, self).__init__()
        self.direction = direction
        # self.a_path = join(image_dir, "a")
        # self.b_path = join(image_dir, "b")
        # self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.root = data_dir
        self.image_filenames = open(filename).readlines()
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a_name, b_name = self.image_filenames[index].split('\n')[0].split("\t")[0], self.image_filenames[index].split('\n')[0].split("\t")[1]
        a = Image.open(join(self.root, a_name)).convert('RGB')
        b = Image.open(join(self.root, b_name)).convert('RGB')
        a = a.resize((286, 286), Image.BICUBIC)
        b = b.resize((286, 286), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b, b_name
        else:
            return b, a, a_name

    def __len__(self):
        return len(self.image_filenames)
