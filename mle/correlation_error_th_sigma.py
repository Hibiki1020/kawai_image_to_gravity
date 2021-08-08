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
    e_arr = np.arctan2(np.sin(l_arr - o_arr), np.cos(l_arr - o_arr))
    return e_arr

def computeMulSigma(cov):
    return np.sqrt(cov[:, 0, 0]) * np.sqrt(cov[:, 1, 1]) * np.sqrt(cov[:, 2, 2])

def computeMAE(x):
    return np.mean(np.abs(x))

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
print("mul_sigma_arr.mean() = ", mul_sigma_arr.mean())

num_steps = 101
resolution_mul_sigma = 0.00001
list_th_mul_sigma = [step*resolution_mul_sigma for step in range(num_steps)]
list_selected_e_r = []
list_selected_e_p = []
list_selected_v_r = []
list_selected_v_p = []
list_selected_num = []
for th_mul_sigma in list_th_mul_sigma:
    print("-----")
    print("th_mul_sigma = ", th_mul_sigma)
    selected_e_r = np.empty(0)
    selected_e_p = np.empty(0)
    for i in range(len(mul_sigma_arr)):
        if mul_sigma_arr[i] > th_mul_sigma:
            selected_e_r = np.append(selected_e_r, e_r_arr[i])
            selected_e_p = np.append(selected_e_p, e_p_arr[i])
    selected_num = selected_e_r.size
    mae_r = computeMAE(selected_e_r/math.pi*180.0)
    mae_p = computeMAE(selected_e_p/math.pi*180.0)
    var_r = np.var(selected_e_r/math.pi*180.0)
    var_p = np.var(selected_e_p/math.pi*180.0)
    ## print
    print("selected_num = ", selected_num)
    print("MAE: r[deg] = ", mae_r, "p[deg] = ", mae_p)
    print("Var: r[deg^2] = ", var_r, "p[deg^2] = ", var_p)
    ## append
    list_selected_e_r.append(mae_r)
    list_selected_e_p.append(mae_p)
    list_selected_v_r.append(var_r)
    list_selected_v_p.append(var_p)
    list_selected_num.append(selected_num)

## graph
plt.figure()
## MAE
plt.subplot(3, 1, 1)
plt.plot(list_th_mul_sigma, list_selected_e_r, label="Roll")
plt.plot(list_th_mul_sigma, list_selected_e_p, label="Pitch")
plt.legend()
plt.xlabel("Threshold of sigma product")
plt.ylabel("MAE")
plt.title("Variance - Error")
## Var
plt.subplot(3, 1, 2)
plt.plot(list_th_mul_sigma, list_selected_e_r, label="Roll")
plt.plot(list_th_mul_sigma, list_selected_e_p, label="Pitch")
plt.legend()
plt.xlabel("Threshold of sigma product")
plt.ylabel("Var")
plt.title("Variance - ErrorVar")
## Var
plt.subplot(3, 1, 3)
plt.plot(list_th_mul_sigma, list_selected_num)
plt.xlabel("Threshold of sigma product")
plt.ylabel("Number of selected samples")
plt.title("Variance - Selected number")

plt.tight_layout()
plt.show()
