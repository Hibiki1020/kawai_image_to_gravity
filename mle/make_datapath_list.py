import csv
import os

def make_datapath_list(rootpath, csv_name, max_num=-1):
    csvpath = os.path.join(rootpath, csv_name)
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        data_list = []
        for row in reader:
            if (max_num > 0) and (len(data_list) >= max_num):
                break
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
