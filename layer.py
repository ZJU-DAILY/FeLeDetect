import torch
import torch.nn as nn
from utils import *
from torch_scatter import segment_csr
import random


class Aggregator1(nn.Module):
    def __init__(self, t_dim, v_dim, a_dim, t_outdim, v_outdim, a_outdim, act=lambda x:x, cuda=False):

        super(Aggregator1, self).__init__()
        self.t_dim = t_dim
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.t_outdim = t_outdim
        self.v_outdim = v_outdim
        self.a_outdim = a_outdim
        self.cuda = cuda
        self.wv = nn.Parameter(torch.zeros(size=(t_dim, v_dim)))
        self.wt = nn.Parameter(torch.zeros(size=(v_dim, t_dim)))
        self.wa_t = nn.Parameter(torch.zeros(size=(v_dim, a_dim)))   #update vnode
        self.w1 = nn.Parameter(torch.zeros(size=(t_outdim, t_dim*2)))
        self.w2 = nn.Parameter(torch.zeros(size=(v_outdim, v_dim*2)))
        self.wa = nn.Parameter(torch.zeros(size=(a_dim, a_outdim)))
        self.wa_v = nn.Parameter(torch.zeros(size=(t_dim, a_dim)))
        self.act = act

        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wt)
        nn.init.xavier_uniform_(self.wa_v)
        nn.init.xavier_uniform_(self.wa_t)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.wa)

    def forward(self, t_info, v_info, t_embed, v_embed, a_embed, name):#adj_matrix :tensnor
        ptr_t, a_list, v_list = t_info
        ptr_t = ptr_t.to(self.cuda)
        a_list = a_list.to(self.cuda)
        v_list = v_list.to(self.cuda)
        a_v = a_embed[a_list] * v_embed[v_list]
        agg_embed = self.wa_v.mm(a_v.t()).t()

        out = segment_csr(agg_embed, ptr_t, reduce=name)
        t_updatembed = self.act(self.w1.mm(torch.cat([t_embed, out], dim=1).t()))
        t_updatembed = t_updatembed.t()

        ptr_v, a_list, t_list = v_info
        ptr_v = ptr_v.to(self.cuda)
        a_list = a_list.to(self.cuda)
        t_list = t_list.to(self.cuda)
        tuple_embed = self.wt.mm(t_embed[t_list].t())
        att_embed2 = self.wa_t.mm(a_embed[a_list].t())
        agg_embed2 = (att_embed2 * tuple_embed).t()
        des = torch.zeros(v_embed.shape[0], agg_embed2.shape[1]).to(self.cuda)
        out2 = segment_csr(agg_embed2, ptr_v, out=des, reduce=name)
        v_updatembed = self.act(self.w2.mm(torch.cat([v_embed, out2], dim=1).t()))
        v_updatembed = v_updatembed.t()
        return t_updatembed, v_updatembed, torch.matmul(a_embed, self.wa)


class Aggregator2(nn.Module):
    def __init__(self, t_dim, v_dim, a_dim, t_outdim, v_outdim, a_outdim, act=lambda x:x, cuda=False):

        super(Aggregator2, self).__init__()
        self.t_dim = t_dim
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.t_outdim = t_outdim
        self.v_outdim = v_outdim
        self.a_outdim = a_outdim
        self.cuda = cuda
        self.wv = nn.Parameter(torch.zeros(size=(t_dim, v_dim)))
        self.wt = nn.Parameter(torch.zeros(size=(v_dim, t_dim)))
        self.wa_t = nn.Parameter(torch.zeros(size=(v_dim, a_dim)))
        self.w2 = nn.Parameter(torch.zeros(size=(v_outdim, v_dim*2)))
        self.wa = nn.Parameter(torch.zeros(size=(a_dim, a_outdim)))
        self.wa_v = nn.Parameter(torch.zeros(size=(t_dim, a_dim)))
        self.act = act

        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wt)
        nn.init.xavier_uniform_(self.wa_v)
        nn.init.xavier_uniform_(self.wa_t)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.wa)

    def forward(self, t_info, v_info, t_embed, v_embed, a_embed, name):#adj_matrix :tensnor
        ptr_t, a_list, v_list = t_info
        ptr_t = ptr_t.to(self.cuda)
        a_list = a_list.to(self.cuda)
        v_list = v_list.to(self.cuda)

        value_embed = self.wv.mm(v_embed[v_list].t())
        att_embed = self.wa_v.mm(a_embed[a_list].t())
        agg_embed = (att_embed * value_embed).t()
        out = segment_csr(agg_embed, ptr_t, reduce=name)
        t_updatembed = self.act(self.w1.mm(torch.cat([t_embed, out], dim=1).t()))
        t_updatembed = t_updatembed.t()

        ptr_v, a_list, t_list = v_info
        ptr_v = ptr_v.to(self.cuda)
        a_list = a_list.to(self.cuda)
        t_list = t_list.to(self.cuda)
        tuple_embed = self.wt.mm(t_embed[t_list].t())
        att_embed2 = self.wa_t.mm(a_embed[a_list].t())
        agg_embed2 = (att_embed2 * tuple_embed).t()
        out2 = segment_csr(agg_embed2, ptr_v,reduce=name)
        v_updatembed = self.act(self.w2.mm(torch.cat([v_embed, out2], dim=1).t()))
        v_updatembed = v_updatembed.t()
        return t_updatembed, v_updatembed, torch.matmul(a_embed, self.wa)

class Linear(nn.Module):
    def __init__(self, input_size, hidden_size, p=0.3):
        super(Linear, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.predict = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        out = self.hidden(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.predict(out)
        return out


