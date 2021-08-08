import torch.utils.data as data
from PIL import Image
import numpy as np

import torch

import make_datapath_list
import data_transform

class OriginalDataset(data.Dataset):
    def __init__(self, data_list, transform, phase):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][0]
        img = Image.open(img_path).convert('RGB')
        #img = img.convert("RGB")

        acc_str_list = self.data_list[index][8:]
        print(self.data_list[index])
        print(acc_str_list)
        acc_list = [float(num) for num in acc_str_list]
        acc = np.array(acc_list)

        img_trans, acc_trans = self.transform(img, acc, phase=self.phase)

        return img_trans, acc_trans

##### test #####
# ## list
# train_rootpath = "../dataset/train"
# val_rootpath = "../dataset/val"
# csv_name = "imu_camera.csv"
# train_list = make_datapath_list.make_datapath_list(train_rootpath, csv_name)
# val_list = make_datapath_list.make_datapath_list(val_rootpath, csv_name)
# ## trans param
# size = 224  #VGG16
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## dataset
# train_dataset = OriginalDataset(
#     data_list=train_list,
#     transform=data_transform.data_transform(size, mean, std),
#     phase="train"
# )
# val_dataset = OriginalDataset(
#     data_list=val_list,
#     transform=data_transform.data_transform(size, mean, std),
#     phase="val"
# )
# ## print
# index = 0
# print("index", index, ": ", train_dataset.__getitem__(index)[0].size())   #data
# print("index", index, ": ", train_dataset.__getitem__(index)[1])   #label
