import csv
from random import random
import torch
import numpy as np

def save_model(model, folder_name, version):
    torch.save(model.state_dict(),
               (folder_name + version + "trained.pth"))


def get_triples(data, v2id):
    triples=[]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            triples.append((i, j, v2id[data[i][j]]))
    return triples


def attribute2id(att):
    a2id = {}
    for r in att[0, :]:
        a2id[r] = len(a2id)
    return a2id


def get_graph(triples):
    tuples = []
    values = []
    edge_type = []
    for t in triples:
        tuples.append(t[0])
        values.append(t[2])
        edge_type.append(t[1])
    return torch.LongTensor([tuples, values]), torch.LongTensor(edge_type)


class Info(object):
    def __init__(self, t, a, v):
        self.at = (a, t)
        self.v = v


def get_index(triples, t_dim, v_dim):
    tuples = []
    values = []
    edge_type = []
    for t in triples:
        tuples.append(t[0])
        edge_type.append(t[1])
        values.append(t[2])

    dict = {i: 0 for i in range(t_dim)}
    for index in tuples:
        dict[int(index)] += 1
    ptr_t = []
    ptr_t.append(0)
    cnt = 0
    for k in dict:
        cnt += dict[k]
        ptr_t.append(cnt)
    t_info = (torch.tensor(ptr_t), torch.tensor(edge_type), torch.tensor(values))

    information = [Info(t,a,v) for (t,a,v) in triples]
    information.sort(key=lambda x: x.v, reverse=False)
    v_index = []
    edge_type = []
    tuples = []
    for info in information:
        v_index.append(info.v)
        edge_type.append(info.at[0])
        tuples.append(info.at[1])
    dict = {i: 0 for i in range(v_dim)}
    for index in v_index:
        dict[int(index)] += 1
    ptr_v = []
    ptr_v.append(0)
    cnt = 0
    for k in dict:
        cnt += dict[k]
        ptr_v.append(cnt)
    v_info = (torch.tensor(ptr_v), torch.tensor(edge_type), torch.tensor(tuples))
    return (t_info, v_info)


def init_embeddings(value_file, att_file):
    value_embddings, att_embeddings = [], []
    with open(value_file) as f:
        for line in f:
            value_embddings.append([float(val) for val in line.strip().split()])

    with open(att_file) as f:
        for line in f:
            att_embeddings.append([float(val) for val in line.strip().split()])
    print("Initialize using bert model.")
    return np.array(value_embddings, dtype=np.float32), np.array(att_embeddings, dtype=np.float32)


def set_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True