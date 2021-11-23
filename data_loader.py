# -*- coding:utf-8 -*-
# Author : yxguo
# Data : 2021/5/10 18:41
from torch.utils.data import Dataset
import torch
import numpy as np
import random

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)


class TrainDataset(Dataset):
    def __init__(self, triples, labels):
        self.triples = triples
        self.labels = labels

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        triple = self.triples[index]
        label = self.labels[index]
        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return triple, trp_label

    def get_neg_ent(self, triple, label):
        def get(triple, label):
            pos_obj = label
            mask = np.ones([self.p.num_ent], dtype=np.bool)
            mask[label] = 0
            neg_ent = np.int32(
                np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
            neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))

            return neg_ent

        neg_ent = get(triple, label)
        return neg_ent


class TestDataset(Dataset):

    def __init__(self, triples, labels):
        self.triples = triples
        self.labels = labels

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        triple = self.triples[index]
        label = self.labels[index]
        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)


class ValidDataset(Dataset):

    def __init__(self, triples, labels):
        self.triples = triples
        self.labels = labels

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        triple = self.triples[index]
        label = self.labels[index]
        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)