import csv
import os

def make_datapath_list(rootpath, csv_name):
    csvpath = os.path.join(rootpath, csv_name)
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        data_list = []
        for row in reader:
            # print(row)
            # print(row[:3])
            # print(row[3])
            row[3] = os.path.join(rootpath, row[3])
            data_list.append(row)
    return data_list

##### test #####
# rootpath = "../dataset/train"
# csv_name = "imu_camera.csv"
# train_list = make_datapath_list(rootpath, csv_name)
# val_list = make_datapath_list(rootpath, csv_name)
# # print(train_list)
# print("example0: ", train_list[0][:3], train_list[0][3])
# print("example1: ", train_list[1][:3], train_list[1][3])
