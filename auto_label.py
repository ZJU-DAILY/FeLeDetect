# -*- coding:utf-8 -*-
# Author : yxguo
# Data : 2021/5/25 20:39
import argparse
import csv
import pickle
import random
import socket
import struct
import sys
import multiprocessing as mp
import time
import numpy as np
from cipher import Relation
from raha.detection import Detection
from raha import dataset


def set_rand_seed(seed=1):
    print("auto.py Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
set_rand_seed(1) #seed = 4 for flights



def party1_al(data_path, args, cat_thre=50, num_thre=0.5):

    relation = Relation(data_path, cat_thre, num_thre)
    sample_index, unsample_index, sample_data = relation.sample_tuple(args.sample_num)  # sample tuples from the whole table
    sample_enc_data = relation.enc()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "localhost"
    try:
        s.connect((host, 8905))
    except Exception:
        print('[!] Server not found or not open')
        sys.exit()

    #  send sample index
    index = pickle.dumps(sample_index)
    header = struct.pack('i', len(index))
    s.send(header)
    s.sendall(index)

    #  send unsample index
    unindex = pickle.dumps(unsample_index)
    header = struct.pack('i', len(unindex))
    s.send(header)
    s.sendall(unindex)

    with open(data_path+'attribute.csv', 'r', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        att = [row for row in reader]

    sample_private_relation = np.vstack((att, sample_enc_data))
    data = pickle.dumps(sample_private_relation)
    header = struct.pack('i', len(data))
    s.send(header)
    s.sendall(data)

    max_size = 15000

    # recv detected error
    header_struct = s.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    detect_recv = pickle.loads(recv)


    num = int(args.sample_num * 6/10)

    train_detect={}
    valid_detect={}
    for k in detect_recv:
        i, j = k
        if i < num:
            train_detect[k] = detect_recv[k]
        else:
            valid_detect[k] = detect_recv[k]
    with open(data_path + 'train_has_label.csv', "w") as file:
        for k in train_detect:
            file.write(str(k[0]) + " " + str(k[1]) + " " + str(train_detect[k]) + "\n")
    with open(data_path + 'valid_has_label.csv', "w") as file:
        for k in valid_detect:
            file.write(str(k[0]) + " " + str(k[1]) + " " + str(valid_detect[k]) + "\n")

    train = sample_data[:num]
    valid = sample_data[num:]
    unsample_data = relation.original_data[unsample_index]

    with open(data_path + 'train.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(train)
    with open(data_path + 'valid.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(valid)
    with open(data_path + 'unsample_data.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(unsample_data)

    sample_label = relation.original_data_labels[sample_index]
    unsample_label = relation.original_data_labels[unsample_index]
    with open(data_path + 'sample_label.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sample_label)
    with open(data_path + 'unsample_label.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(unsample_label)


    # with open(data_path + 'test_label.csv', 'w', encoding='UTF-8', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(test_label)


def party2_al(data_path, args):

    relation = Relation(data_path)
    host = "localhost"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, 8905))
    s.listen(5)
    print("listening...")
    conn, addr = s.accept()
    print('[+] Connected with', addr) 

    max_size = 15000

    ##  接收 sample index
    header_struct = conn.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    sample_index = pickle.loads(recv)

    ##  recv unsample index
    header_struct = conn.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    unsample_index = pickle.loads(recv)

    ##  recv sample private data
    header_struct = conn.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        recv += x
        need_recv_size -= len(x)
    private_relation_recv = pickle.loads(recv)

    # gen raha data
    with open(data_path+'attribute.csv', 'r', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        att = [row for row in reader]
    my_sample_data = relation.original_data[sample_index]
    my_sample_relation = np.vstack((att, my_sample_data))
    merge_sample_relation = np.hstack((private_relation_recv, my_sample_relation))  # to label(detect)
    with open("./datasets/raha_datasets/{}/dirty.csv".format(args.dataset), 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(merge_sample_relation)

    my_sample_relation_gt = np.vstack((att, relation.original_data_gt[sample_index]))
    with open('./datasets/distri_datasets/{}/1/sample_enc_data_gt.csv'.format(args.dataset), 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        op_data_ground_truth = [row for row in reader]
        op_data_ground_truth = np.array(op_data_ground_truth)
        op_sample_relation_gt = np.vstack((private_relation_recv[0], op_data_ground_truth))
    merge_sample_relation_gt = np.hstack((op_sample_relation_gt, my_sample_relation_gt))
    with open("./datasets/raha_datasets/{}/clean.csv".format(args.dataset), 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(merge_sample_relation_gt)

    # run raha
    dataset_name =args.dataset
    dataset_dictionary = {
        "name": dataset_name,
        "path": "./datasets/raha_datasets/{}/dirty.csv".format(args.dataset),
        "clean_path": "./datasets/raha_datasets/{}/clean.csv".format(args.dataset)
    }
    app = Detection()
    detection_dictionary = app.run(dataset_dictionary)

    data = dataset.Dataset(dataset_dictionary)
    op_col_num = private_relation_recv.shape[1]
    op_detect = {}
    my_detect = {}
    for key in detection_dictionary:
        i, j = key
        if j < op_col_num:
            op_detect[key] = detection_dictionary[key]
        else:
            my_detect[(i, j-op_col_num)] = detection_dictionary[key]

    # send op_detect
    op_detect_send = pickle.dumps(op_detect)
    header = struct.pack('i', len(op_detect_send))
    conn.send(header)
    conn.sendall(op_detect_send)

    num = int(args.sample_num * 6/10)
    train = my_sample_data[:num]
    valid = my_sample_data[num:]
    unsample_data = relation.original_data[unsample_index]

    with open(data_path + 'train.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(train)
    with open(data_path + 'valid.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(valid)
    with open(data_path + 'unsample_data.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(unsample_data)
    train_detect = {}
    valid_detect = {}
    for k in my_detect:
        if k[0] < num:
            train_detect[k] = my_detect[k]
        else:
            valid_detect[k] = my_detect[k]
    with open(data_path + 'train_has_label.csv', "w") as file:
        for k in train_detect:
            file.write(str(k[0]) + " " + str(k[1]) + " " + str(train_detect[k]) + "\n")
    with open(data_path + 'valid_has_label.csv', "w") as file:
        for k in valid_detect:
            file.write(str(k[0]) + " " + str(k[1]) + " " + str(valid_detect[k]) + "\n")
    sample_label = relation.original_data_labels[sample_index]
    unsample_label = relation.original_data_labels[unsample_index]
    with open(data_path + 'sample_label.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sample_label)
    with open(data_path + 'unsample_label.csv', 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(unsample_label)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-dataset', default="Adult_5",
                      help='choose dataset')
    args.add_argument('-sample_num', type=int, default = 450,
                      help='sample some tuples to be auto-labeld')
    args = args.parse_args()
    return args


def merge(name):
    with open("./datasets/distri_datasets/{}/1/{}.csv".format(args.dataset,name),'r',encoding='UTF-8') as p1:
        reader = csv.reader(p1)
        x1 = [row for row in reader]
        x1 = np.array(x1)
    with open("./datasets/distri_datasets/{}/2/{}.csv".format(args.dataset,name),'r',encoding='UTF-8') as p2:
        reader = csv.reader(p2)
        x2 = [row for row in reader]
        x2 = np.array(x2)
    x = np.hstack((x1, x2))

    with open('./datasets/centra_datasets/{}/{}.csv'.format(args.dataset,name),'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(x)


if __name__ == '__main__':
    args = parse_args()
    set_rand_seed(1)
    p2 = mp.Process(target=party2_al,
                    args=("./datasets/distri_datasets/{}/2/".format(args.dataset), args))
    set_rand_seed(1)
    p1 = mp.Process(target=party1_al,
                    args=("./datasets/distri_datasets/{}/1/".format(args.dataset), args))

    p2.start()
    time.sleep(5)
    p1.start()

    p2.join()
    p1.join()

    merge("train")
    merge("valid")
    merge("dirty")
    merge("clean")
    merge("sample_label")
    merge("unsample_data")
    merge("unsample_label")
    with open("./datasets/distri_datasets/{}/1/train_has_label.csv".format(args.dataset), 'r', encoding='UTF-8') as p1:
        reader = csv.reader(p1)
        x1 = [row for row in reader]
        x1 = np.array(x1)
    with open("./datasets/distri_datasets/{}/2/train_has_label.csv".format(args.dataset), 'r', encoding='UTF-8') as p2:
        reader = csv.reader(p2)
        x2 = [row for row in reader]
        x2 = np.array(x2)
    with open('./datasets/centra_datasets/{}/train_has_label.csv'.format(args.dataset), 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(x1)

    with open('./datasets/centra_datasets/{}/train_has_label.csv'.format(args.dataset), 'a+', encoding='UTF-8', newline='') as csvfile:
        for r in x2:
            i, j, l= "".join(r).split()
            i = int(i)
            j = int(j) + 4
            l = int(l)
            csvfile.write(str(i) + " " + str(j) + " " + str(l) + "\n")

    with open("./datasets/distri_datasets/{}/1/valid_has_label.csv".format(args.dataset), 'r', encoding='UTF-8') as p1:
        reader = csv.reader(p1)
        x1 = [row for row in reader]
        x1 = np.array(x1)
    with open("./datasets/distri_datasets/{}/2/valid_has_label.csv".format(args.dataset), 'r', encoding='UTF-8') as p2:
        reader = csv.reader(p2)
        x2 = [row for row in reader]
        x2 = np.array(x2)
    with open('./datasets/centra_datasets/{}/valid_has_label.csv'.format(args.dataset), 'w', encoding='UTF-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(x1)

    with open('./datasets/centra_datasets/{}/valid_has_label.csv'.format(args.dataset), 'a+', encoding='UTF-8', newline='') as csvfile:
        for r in x2:
            i, j, l = "".join(r).split()
            i = int(i)
            j = int(j) + 4
            l = int(l)
            csvfile.write(str(i) + " " + str(j) + " " + str(l) + "\n")
