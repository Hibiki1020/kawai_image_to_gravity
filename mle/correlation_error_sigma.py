import matplotlib.pyplot as plt
import numpy as np
import math

import torch
from torchvision import models
import torch.nn as nn

import make_datapath_list
import data_transform
import original_dataset
import original_network
import original_criterion

## device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)

## network
net = original_network.OriginalNet()
print(net)
net.to(device)
net.eval()

## saved in CPU -> load in CPU, saved in GPU -> load in GPU
load_path = "../weights/mle.pth"
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)

## trans param
size = 224  #VGG16
mean_element = 0.5
std_element = 0.5
mean = ([mean_element, mean_element, mean_element])
std = ([std_element, std_element, std_element])

## list
# rootpath = "../dataset/train"
rootpath = "../dataset/val"
csv_name = "imu_camera.csv"
val_list = make_datapath_list.make_datapath_list(rootpath, csv_name)

## transform
transform = data_transform.data_transform(size, mean, std)

## dataset
val_dataset = original_dataset.OriginalDataset(
    data_list=val_list,
    transform=data_transform.data_transform(size, mean, std),
    phase="val"
)

## dataloader
batch_size = 25
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

## mini-batch prediction
labels_arr = np.empty([0, 3])
mu_arr = np.empty([0, 3])
cov_arr = np.empty([0, 3, 3])
for inputs, labels in val_dataloader:
    inputs = inputs.to(device)
    outputs = net(inputs)
    Cov = original_criterion.getCovMatrix(outputs)
    labels_arr = np.append(labels_arr, labels.cpu().detach().numpy(), axis=0)
    mu_arr = np.append(mu_arr, outputs[:, :3].cpu().detach().numpy(), axis=0)
    cov_arr = np.append(cov_arr, Cov.cpu().detach().numpy(), axis=0)

print("labels_arr.shape = ", labels_arr.shape)
print("mu_arr.shape = ", mu_arr.shape)
print("cov_arr.shape = ", cov_arr.shape)

def accToRP(acc):
    r = np.arctan2(acc[:, 1], acc[:, 2])
    p = np.arctan2(-acc[:, 0], np.sqrt(acc[:, 1]*acc[:, 1] + acc[:, 2]*acc[:, 2]))
    return r, p

def computeError(l_arr, o_arr):
    return np.arctan2(np.sin(l_arr - o_arr), np.cos(l_arr - o_arr))

def computeMulSigma(cov):
    return np.sqrt(cov[:, 0, 0]) * np.sqrt(cov[:, 1, 1]) * np.sqrt(cov[:, 2, 2])

l_r_arr, l_p_arr = accToRP(labels_arr)
o_r_arr, o_p_arr = accToRP(mu_arr)
e_r_arr = computeError(l_r_arr, o_r_arr)
e_p_arr = computeError(l_p_arr, o_p_arr)
mul_sigma_arr = computeMulSigma(cov_arr)
print("l_r_arr.shape = ", l_r_arr.shape)
print("l_p_arr.shape = ", l_p_arr.shape)
print("o_r_arr.shape = ", o_r_arr.shape)
print("o_p_arr.shape = ", o_p_arr.shape)
print("e_r_arr.shape = ", e_r_arr.shape)
print("e_p_arr.shape = ", e_p_arr.shape)
print("mul_sigma_arr.shape = ", mul_sigma_arr.shape)

## graph
plt.figure()
plt.scatter(mul_sigma_arr, np.abs(e_r_arr/math.pi*180.0), label="Roll")
plt.scatter(mul_sigma_arr, np.abs(e_p_arr/math.pi*180.0), label="Pitch")
plt.legend()
plt.xlabel("Sigma product")
plt.ylabel("AE")
plt.title("Sigma - Error")
plt.xlim(min(mul_sigma_arr), max(mul_sigma_arr))
plt.ylim(min(np.abs(e_r_arr/math.pi*180.0)), max(np.abs(e_r_arr/math.pi*180.0)))

plt.tight_layout()
plt.show()
