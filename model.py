from layer import *
import torch
import torch.nn as nn
import numpy as np
from utils import *


class Model(nn.Module):
    def __init__(self, t_embed, v_embed, a_embed, params, cuda):
        super(Model, self).__init__()
        self.p = params
        self.t_dim = params.tuple_init_dim
        self.v_dim = params.value_init_dim
        self.a_dim = params.att_init_dim
        self.t_init_embeddings = t_embed
        self.v_init_embeddings = v_embed
        self.a_init_embeddings = a_embed
        self.act = torch.tanh
        self.cuda = cuda
        self.layer1 = Aggregator2(self.t_dim, self.v_dim, self.a_dim, self.p.t_outdim1, self.p.v_outdim1,
                                 self.p.a_outdim1, self.act, cuda=self.cuda)
        self.layer2 = Aggregator2(self.layer1.t_outdim, self.layer1.v_outdim, self.layer1.a_outdim, self.p.t_outdim2,
                                 self.p.v_outdim2, self.p.a_outdim2, self.act, cuda=self.cuda)

        triembed_dim = self.p.t_outdim2 + self.p.v_outdim2 + self.p.a_outdim2
        self.linear = Linear(triembed_dim, self.p.hid_dim, self.p.dropout)

    def forward(self, t_info, v_info, name, batch):
        t_embed1, v_embed1, a_embed1 = self.layer1(t_info, v_info, self.t_init_embeddings, self.v_init_embeddings,
                                                   self.a_init_embeddings, name)
        t_embed2, v_embed2, a_embed2 = self.layer2(t_info, v_info, t_embed1, v_embed1, a_embed1, name)

        t = t_embed2[batch[:, 0]]
        a = a_embed2[batch[:, 1]]
        v = v_embed2[batch[:, 2]]
        x = torch.cat([t, a, v], dim=1)
        score = self.linear(x)  #分类器
        return score


class Classifier(nn.Module):
    def __init__(self, params):
        super(Classifier, self).__init__()
        self.p = params
        triembed_dim = self.p.t_outdim2 + self.p.v_outdim2 + self.p.a_outdim2
        self.linear = Linear(triembed_dim, self.p.hid_dim, self.p.dropout)

    def forward(self, t_embed, v_embed, a_embed, batch):
        t = t_embed[batch[:, 0]]
        a = a_embed[batch[:, 1]]
        v = v_embed[batch[:, 2]]
        x = torch.cat([t, a, v], dim=1)
        score = self.linear(x)
        return score