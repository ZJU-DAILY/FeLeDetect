from fed_layer import *
import torch
import torch.nn as nn


class Fed_Model1(nn.Module):
    def __init__(self, t_embed, v_embed, a_embed, params, cuda, comp=32, tau=0):
        super(Fed_Model1, self).__init__()
        self.p = params
        self.t_dim = params.tuple_init_dim
        self.v_dim = params.value_init_dim
        self.a_dim = params.att_init_dim
        self.t_init_embeddings = t_embed
        self.v_init_embeddings = v_embed
        self.a_init_embeddings = a_embed
        self.cuda = cuda
        self.comp = comp
        self.tau = tau
        self.act = torch.tanh
        self.layer1 = Aggregator1_(self.t_dim, self.v_dim, self.a_dim, self.p.t_outdim1, self.p.v_outdim1,
                                     self.p.a_outdim1, self.act, torch.device("cuda:0")).to('cuda:0')
        self.layer2 = Aggregator2_1(self.layer1.t_outdim, self.layer1.v_outdim, self.layer1.a_outdim, self.p.t_outdim2,
                                    self.p.v_outdim2, self.p.a_outdim2, self.act, torch.device("cuda:1"), self.comp, self.tau).to('cuda:1')
        triembed_dim = self.p.t_outdim2 + self.p.v_outdim2 + self.p.a_outdim2
        self.linear = Linear(triembed_dim, self.p.hid_dim, self.p.dropout).to('cuda:1')


    def forward(self, t_info, v_info, name, batch, a, v, conn,  v_index_recv='',v_id_list='', train =False):

        t_embed1, v_embed1, a_embed1 = self.layer1(t_info, v_info, self.t_init_embeddings.to('cuda:0'), self.v_init_embeddings.to('cuda:0'),
                                                   self.a_init_embeddings.to('cuda:0'), a.to('cuda:0'), v.to('cuda:0'), name, conn)
        '''layer2'''
        t_embed2, v_embed2, a_embed2 = self.layer2(t_info, v_info, t_embed1.to('cuda:1'), v_embed1.to('cuda:1'), a_embed1.to('cuda:1'),
                                                   name, conn, v_index_recv, v_id_list, train)

        t = t_embed2[batch[:, 0]]
        a = a_embed2[batch[:, 1]]
        v = v_embed2[batch[:, 2]]
        x = torch.cat([t, a, v], dim=1)
        score = self.linear(x)
        return score


class Fed_Model2(nn.Module):
    def __init__(self, t_embed, v_embed, a_embed, params,cuda,comp=32,tau =0):
        super(Fed_Model2, self).__init__()
        self.p = params
        self.t_dim = params.tuple_init_dim
        self.v_dim = params.value_init_dim
        self.a_dim = params.att_init_dim
        self.t_init_embeddings = t_embed
        self.v_init_embeddings = v_embed
        self.a_init_embeddings = a_embed
        self.cuda = cuda
        self.comp = comp
        self.tau = tau
        self.act = torch.tanh

        self.layer1 = Aggregator1_(self.t_dim, self.v_dim, self.a_dim, self.p.t_outdim1, self.p.v_outdim1,
                                     self.p.a_outdim1, self.act, torch.device("cuda:2")).to("cuda:2")
        self.layer2 = Aggregator2_2(self.layer1.t_outdim, self.layer1.v_outdim, self.layer1.a_outdim, self.p.t_outdim2,
                                    self.p.v_outdim2, self.p.a_outdim2,  self.act, torch.device("cuda:3"), self.comp, self.tau).to("cuda:3")
        triembed_dim = self.p.t_outdim2 + self.p.v_outdim2 + self.p.a_outdim2
        self.linear = Linear(triembed_dim, self.p.hid_dim, self.p.dropout).to("cuda:3")


    def forward(self,  t_info, v_info, name, batch, a,v, conn,  v_index_recv='', v_id_list='', train=False):


        t_embed1, v_embed1, a_embed1 = self.layer1(t_info, v_info, self.t_init_embeddings.to("cuda:2"), self.v_init_embeddings.to("cuda:2"),
                                                       self.a_init_embeddings.to("cuda:2"), a.to("cuda:2"), v.to("cuda:2"), name, conn)

        '''layer2'''
        t_embed2, v_embed2, a_embed2 = self.layer2(t_info, v_info, t_embed1.to("cuda:3"), v_embed1.to("cuda:3"), a_embed1.to("cuda:3"), name, conn, v_index_recv, v_id_list, train)
        t = t_embed2[batch[:, 0]]
        a = a_embed2[batch[:, 1]]
        v = v_embed2[batch[:, 2]]
        x = torch.cat([t, a, v], dim=1)
        score = self.linear(x)
        return score
