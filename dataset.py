import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def get_image_and_label(input_dir):
    x_list = []
    y_list = []
    dir_list = os.listdir(input_dir)
    for i, dir_name in enumerate(dir_list):
        file_list = glob.glob(input_dir + dir_name + '/*')
        tmp_y_list = [np.int64(i)] * len(file_list)
        x_list.extend(file_list)
        y_list.extend(tmp_y_list)
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y


class MyDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        image = image.transpose(2, 0, 1)

        label = self.y[idx] if self.y is not None else None
        if label is not None:
            return image, label
        else:
            return image


class TestDataset(Dataset):
    def __init__(self, x, transform, test_path):
        self.x = x
        self.transform = transform
        self.test_path = test_path

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        file_name = self.x[idx]
        # file_path = f'{self.test_path}/{file_name}'
        file_path = f'{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        image = image.transpose(2, 0, 1)
        return image
