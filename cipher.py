# -*- coding:utf-8 -*-
# Author : yxguo
# Data : 2021/5/25 21:37
# -*- coding:utf-8 -*-
# Author : yxguo
# Data : 2021/5/25 10:33
import csv
import random

import numpy as np
import copy


def set_rand_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
set_rand_seed(1)


def kaisa(str):
    key = 13
    symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 !?.,;#$%^&*_+=:'
    ciphers = symbols[int(key):] + symbols[:int(key)]
    transtab = str.maketrans(symbols, ciphers)

    message = str
    result = message.translate(transtab)
    return result


class Relation(object):
    def __init__(self, data_path, cat_thre = 1000, num_thre=0.5):
        with open((data_path + 'dirty.csv'), 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            original_data = [row for row in reader]
            original_data = np.array(original_data)
            original_data = np.delete(original_data, 0, axis=0)
        with open((data_path + 'clean.csv'), 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            original_data_gt = [row for row in reader]
            original_data_gt = np.array(original_data_gt)
            original_data_gt = np.delete(original_data_gt, 0, axis=0)
        with open((data_path + 'labels.csv'), 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            original_data_labels = [row for row in reader]
            original_data_labels = np.array(original_data_labels)

        self.path = data_path
        self.original_data = original_data
        self.original_data_gt = original_data_gt
        self.original_data_labels = original_data_labels
        self.cat_threshold = cat_thre
        self.num_threshold = num_thre
        self.types = self.getTypes()
        self.domains = self.getDomains()
        self.change = []
        print("type", self.types)


    def sample_tuple(self, sample_num):
        set_rand_seed(1)
        rand_arr = np.arange(self.original_data.shape[0])
        np.random.shuffle(rand_arr)
        sample_indices = rand_arr[:sample_num]
        unsample_indices = rand_arr[sample_num:]
        sample_data = self.original_data[sample_indices]
        self.sample_data = self.original_data[sample_indices]
        self.sample_data_gt = self.original_data_gt[sample_indices]
        return sample_indices, unsample_indices, sample_data


    def enc(self):
        sample_enc_data = copy.copy(self.sample_data)
        for i in range(sample_enc_data.shape[0]):
            for j in range(sample_enc_data.shape[1]):
                sample_enc_data[i][j] = kaisa(sample_enc_data[i][j])

        sample_enc_data_gt = copy.copy(self.sample_data_gt)
        for i in range(sample_enc_data_gt.shape[0]):
            for j in range(sample_enc_data_gt.shape[1]):
                sample_enc_data_gt[i][j] = kaisa(sample_enc_data_gt[i][j])
        with open((self.path + 'sample_enc_data_gt.csv'), 'w', encoding='UTF-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(sample_enc_data_gt)
        return sample_enc_data


    def getDomains(self):

        col_num = self.original_data.shape[1]
        domains = {i: set() for i in range(col_num)}
        for datrow in self.original_data:
            for i in range(col_num):
                domains[i].add(datrow[i])
        return domains


    def getTypes(self):
        col_num = self.original_data.shape[1]
        type_list = []
        for i in range(col_num):
            if self.is_cat(self.original_data, i):
                type_list.append('categorical')
            elif self.is_num(self.original_data, i):
                type_list.append('numerical')
                # print i, [d[i] for d in data]
            else:
                type_list.append('string')
        return type_list


    def is_num(self, data, i):
        numerical_count = 0.0
        for datrow in data:
            try:
                float(datrow[i].strip())
                numerical_count = numerical_count + 1.0
            except:
                pass
        return (numerical_count / len(data) > self.num_threshold)


    def is_cat(self, data, i):
        v_set = {}
        for datrow in data:
            if datrow[i] not in v_set:
                v_set[datrow[i]] = 1
            else:
                v_set[datrow[i]] += 1
        return len(v_set) < 50

