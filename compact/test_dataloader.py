import numpy as np
import matplotlib.pyplot as plt

import torch

import make_datapath_list
import data_transform
import original_dataset

def show_inputs(inputs):
    h = 4
    w = 8
    i = 0
    plt.figure()
    for tensor in inputs:
        # print(tensor.size())
        img = tensor.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.subplot(h, w, i+1)
        plt.imshow(img)
        i += 1
    plt.show()

## list
train_rootpath = "../dataset/train"
val_rootpath = "../dataset/val"
csv_name = "imu_camera.csv"
train_list = make_datapath_list.make_datapath_list(train_rootpath, csv_name)
val_list = make_datapath_list.make_datapath_list(val_rootpath, csv_name)

## trans param
size = 224  #VGG16
mean = ([0.5, 0.5, 0.5])
std = ([0.5, 0.5, 0.5])

## dataset
train_dataset = original_dataset.OriginalDataset(
    data_list=train_list,
    transform=data_transform.data_transform(size, mean, std),
    phase="train"
)
val_dataset = original_dataset.OriginalDataset(
    data_list=val_list,
    transform=data_transform.data_transform(size, mean, std),
    phase="val"
)

# dataloader
batch_size = 32

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

batch_iterator = iter(dataloaders_dict["train"])
# batch_iterator = iter(dataloaders_dict["val"])
inputs, labels = next(batch_iterator)

print("inputs.size() = ", inputs.size())
print("labels = ", labels)
print("labels[0] = ", labels[0])
print("labels.size() = ", labels.size())
show_inputs(inputs)
