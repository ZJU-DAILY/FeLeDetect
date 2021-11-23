# -*- coding:utf-8 -*-
# Author : yxguo
# Data : 2021/5/4 14:24
import struct
import time

import torch
import torch.nn as nn
from torch_scatter import segment_csr
import random
from sklearn.metrics.pairwise import cosine_similarity
from compress import *

class Aggregator1_(nn.Module):
    def __init__(self, t_dim, v_dim, a_dim, t_outdim, v_outdim, a_outdim, act=lambda x:x, cuda=False):

        super(Aggregator1_, self).__init__()

        self.t_dim = t_dim
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.t_outdim = t_outdim
        self.v_outdim = v_outdim
        self.a_outdim = a_outdim
        self.cuda = cuda
        self.wv = nn.Parameter(torch.zeros(size=(t_dim, v_dim)))
        self.wt = nn.Parameter(torch.zeros(size=(a_dim, t_dim)))
        self.wa_v = nn.Parameter(torch.zeros(size=(t_dim, a_dim)))#update tnode
        self.wa_t = nn.Parameter(torch.zeros(size=(v_dim, a_dim)))#update vnode
        self.w1 = nn.Parameter(torch.zeros(size=(t_outdim, t_dim*2)))
        self.w2 = nn.Parameter(torch.zeros(size=(v_outdim, v_dim*2)))
        self.wa = nn.Parameter(torch.zeros(size=(a_dim, a_outdim )))
        self.act = act

        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wt)
        nn.init.xavier_uniform_(self.wa_v)
        nn.init.xavier_uniform_(self.wa_t)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.wa)


    def forward(self, t_info, v_info, t_embed, v_embed, a_embed, a_recv, v_recv, name, conn):  # adj_matrix :tensnor

        ptr_t, a_list, v_list = t_info
        a_list = a_list.to(self.cuda)
        a_merge = torch.cat([a_embed[a_list], a_recv], dim=0)
        v_list = v_list.to(self.cuda)
        v_merge = torch.cat([v_embed[v_list], v_recv], dim=0)
        att_embed = self.wa_v.mm(a_merge.t())
        value_embed = self.wv.mm(v_merge.t())
        agg_embed = (att_embed * value_embed).t()

        ptr_t = ptr_t.to(self.cuda)
        mat1 = segment_csr(agg_embed[:int(agg_embed.size(0) / 2)], ptr_t, reduce=name)
        mat2 = segment_csr(agg_embed[int(agg_embed.size(0) / 2):], ptr_t, reduce=name)
        out = (mat1 + mat2) / 2

        t_updatembed = self.act(self.w1.mm(torch.cat([t_embed, out], dim=1).t())).t()
        ptr_v, a_list, t_list = v_info
        t_list = t_list.to(self.cuda)
        tuple_embed = self.wt.mm(t_embed[t_list].t())

        a_list = a_list.to(self.cuda)
        att_embed2 = self.wa_t.mm(a_embed[a_list].t())
        agg_embed2 = (att_embed2 * tuple_embed).t()

        ptr_v = ptr_v.to(self.cuda)
        out2 = segment_csr(agg_embed2, ptr_v, reduce=name)
        v_updatembed = self.act(self.w2.mm(torch.cat([v_embed, out2], dim=1).t())).t()


        return t_updatembed, v_updatembed, torch.matmul(a_embed, self.wa)


class Aggregator2_1(nn.Module):
    def __init__(self, t_dim, v_dim, a_dim, t_outdim, v_outdim, a_outdim, act=lambda x:x, cuda=False, comp=32, tau=0):

        super(Aggregator2_1, self).__init__()

        self.t_dim = t_dim
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.t_outdim = t_outdim
        self.v_outdim = v_outdim
        self.a_outdim = a_outdim
        self.cuda = cuda
        self.compress = comp
        self.tau = tau
        self.wv = nn.Parameter(torch.zeros(size=(t_dim, v_dim)))
        self.wt = nn.Parameter(torch.zeros(size=(v_dim, t_dim)))
        self.wa_v = nn.Parameter(torch.zeros(size=(t_dim, a_dim)))
        self.wa_t = nn.Parameter(torch.zeros(size=(v_dim, a_dim)))
        self.w1 = nn.Parameter(torch.zeros(size=(t_outdim, t_dim*2)))
        self.w2 = nn.Parameter(torch.zeros(size=(v_outdim, v_dim*2)))
        self.wa = nn.Parameter(torch.zeros(size=(a_outdim, a_dim)))
        self.w11 = nn.Parameter(torch.zeros(size=(t_dim, t_dim)))
        self.w12 = nn.Parameter(torch.zeros(size=(t_dim, t_dim)))
        self.act = act
        self.last_send_v = np.zeros((1, 1), dtype=np.float32)
        self.last_recv_v = np.zeros((1, 1), dtype=np.float32)
        self.send = 0
        self.send_time = 0.0
        self.exchange_cost = 0.0

        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wt)
        nn.init.xavier_uniform_(self.wa_v)
        nn.init.xavier_uniform_(self.wa_t)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.wa)
        nn.init.xavier_uniform_(self.w11)
        nn.init.xavier_uniform_(self.w12)


    def forward(self, t_info, v_info, t_embed, v_embed, a_embed, name, conn, v_index='', v_id_send='', train=False):#adj_matrix :tensnor
        ptr_t, a_list, v_list = t_info
        ptr_t = ptr_t.to(self.cuda)

        max_size = 0xffffffffffff
        device = self.cuda
        if self.tau == 0:
            if self.compress == 32:
                v = v_embed[v_id_send].cpu().detach().numpy()
                v_bytes = v.tobytes()
                header = struct.pack('i', len(v_bytes))
                self.exchange_cost += len(v_bytes)
                b = time.time()
                conn.sendall(header)
                conn.sendall(v_bytes)
                e = time.time()
                self.send_time += (e - b)

                header_struct = conn.recv(4)
                unpack_res = struct.unpack('i', header_struct)
                need_recv_size = unpack_res[0]
                self.exchange_cost += need_recv_size
                v_embed_recv = b""

                b = time.time()
                while need_recv_size > 0:
                    x = conn.recv(min(max_size, need_recv_size))
                    v_embed_recv += x
                    need_recv_size -= len(x)
                e = time.time()
                self.send_time += (e - b)
                self.send += 1

                v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.float32)).reshape(-1, self.v_dim)
                v_embed_recv = v_embed_recv[v_index]
                v_embed_recv = torch.from_numpy(np.array(np.array(v_embed_recv))).to(device)

            else:
                if self.compress == 16:
                    interval = 2/65535
                elif self.compress == 8:
                    interval = 2/255
                elif self.compress == 4:
                    interval = 2/15
                elif self.compress == 2:
                    interval = 2/3
                else:
                    interval = 2
                v = v_embed[v_id_send].cpu().detach().numpy()
                v_com = compress(v, interval)
                v_bytes = v_com.tobytes()
                header = struct.pack('i', len(v_bytes))
                self.exchange_cost += len(v_bytes)
                b = time.time()
                conn.sendall(header)
                conn.sendall(v_bytes)
                e = time.time()
                self.send_time += (e - b)

                header_struct = conn.recv(4)
                unpack_res = struct.unpack('i', header_struct)
                need_recv_size = unpack_res[0]
                self.exchange_cost += need_recv_size
                v_embed_recv = b""

                b = time.time()
                while need_recv_size > 0:
                    x = conn.recv(min(max_size, need_recv_size))
                    v_embed_recv += x
                    need_recv_size -= len(x)
                e = time.time()
                self.send_time += (e - b)
                self.send += 1
                if self.compress == 8:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, self.v_dim)
                elif self.compress == 16:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint16)).reshape(-1, self.v_dim)
                elif self.compress == 4:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, int(self.v_dim/2))
                elif self.compress == 2:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, int(self.v_dim / 4))
                else:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, self.v_dim)
                v_embed_recv = decompress(v_embed_recv, interval)
                v_embed_recv = v_embed_recv[v_index]
                v_embed_recv = torch.from_numpy(np.array(v_embed_recv)).to(device)
        else:
            interval = 2 / 15
            v = v_embed[v_id_send].cpu().detach().numpy()
            if train and self.last_send_v.shape[0] > 1:
                dist = np.linalg.norm(v - self.last_send_v)
                # cos = cosine_similarity(v, self.last_send_v)   #cos similarity
                if dist <= self.tau:   # dont send
                    header = struct.pack('i', 0)
                    conn.sendall(header)

                else:
                    self.last_send_v = v.copy()
                    v_com = compress(v, interval)
                    v_bytes = v_com.tobytes()
                    header = struct.pack('i', len(v_bytes))
                    b = time.time()
                    conn.sendall(header)
                    conn.sendall(v_bytes)
                    self.send_time += time.time() - b
                    self.send += 1
                    self.exchange_cost += len(v_bytes)
            else:
                v_com = compress(v, interval)
                v_bytes = v_com.tobytes()
                header = struct.pack('i', len(v_bytes))
                b = time.time()
                conn.sendall(header)
                conn.sendall(v_bytes)
                self.exchange_cost += len(v_bytes)
                self.send_time += time.time() - b
                if train: #first forward
                    self.last_send_v = v.copy()

            header_struct = conn.recv(4)
            unpack_res = struct.unpack('i', header_struct)
            need_recv_size = unpack_res[0]
            v_embed_recv = b""
            if need_recv_size > 0:
                self.exchange_cost += need_recv_size
                b = time.time()
                while need_recv_size > 0:
                    x = conn.recv(min(max_size, need_recv_size))
                    v_embed_recv += x
                    need_recv_size -= len(x)
                self.send_time += time.time() - b
                v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, int(self.v_dim/2))
                v_embed_recv = decompress(v_embed_recv, interval)
                v_embed_recv = v_embed_recv[v_index]
                if train:
                    self.last_recv_v = v_embed_recv.copy()
                v_embed_recv = torch.from_numpy(np.array(v_embed_recv)).to(device)
            else:
                v_embed_recv = torch.from_numpy(np.array(self.last_recv_v)).to(device)

            v_list = v_list.to(self.cuda)
            merge_v = torch.cat([v_embed[v_list], v_embed_recv], dim=0)
            a = a_embed.cpu().detach().numpy()
            a_bytes = a.tobytes()
            conn.sendall(a_bytes)

            need_recv_size = len(a_bytes)
            a_embed_recv = b""
            while need_recv_size > 0:
                x = conn.recv(min(150000, need_recv_size))
                a_embed_recv += x
                need_recv_size -= len(x)
            self.exchange_cost += 2*len(a_bytes)

            a_embed_recv = (np.frombuffer(a_embed_recv, dtype=np.float32)).reshape(a.shape[0], a.shape[1])
            a_embed_recv = torch.from_numpy(np.array(a_embed_recv)).to(device)
            a_list = a_list.to(self.cuda)
            a_embed_recv = a_embed_recv[a_list]
            merge_a = torch.cat([a_embed[a_list], a_embed_recv], dim=0)


        value_embed = self.wv.mm(merge_v.t())  # t_dim*v_dim  X v_dim*2n
        att_embed = self.wa_v.mm(merge_a.t())  # t_dim*a_dim  X a_dim*2n
        agg_embed = (att_embed * value_embed).t()  # 2n*tdim
        mat1 = segment_csr(agg_embed[:int(agg_embed.size(0) / 2)], ptr_t, reduce=name)
        mat2 = segment_csr(agg_embed[int(agg_embed.size(0) / 2):], ptr_t, reduce=name)
        out = (mat1 + mat2) / 2
        t_updatembed = self.act(self.w1.mm(torch.cat([t_embed, out], dim=1).t())).t()

        ptr_v, a_list, t_list = v_info
        t_list = t_list.to(self.cuda)
        tuple_embed = self.wt.mm(t_embed[t_list].t())

        a_list = a_list.to(self.cuda)
        att_embed2 = self.wa_t.mm(a_embed[a_list].t())

        agg_embed2 = att_embed2 * tuple_embed
        ptr_v = ptr_v.to(self.cuda)
        out2 = segment_csr(agg_embed2.t(), ptr_v, reduce=name)
        v_updatembed = self.act(self.w2.mm(torch.cat([v_embed, out2], dim=1).t())).t()
        a_updateembed = self.wa.mm(a_embed.t()).t()

        return t_updatembed, v_updatembed, a_updateembed  #torch.matmul(a_embed, self.wa)


class Aggregator2_2(nn.Module):
    def __init__(self, t_dim, v_dim, a_dim, t_outdim, v_outdim, a_outdim, act=lambda x:x, cuda=False, compress=32, tau=0):

        super(Aggregator2_2, self).__init__()

        self.t_dim = t_dim
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.t_outdim = t_outdim
        self.v_outdim = v_outdim
        self.a_outdim = a_outdim
        # self.features = features
        self.cuda = cuda
        self.compress = compress
        self.tau = tau
        self.wv = nn.Parameter(torch.zeros(size=(t_dim, v_dim)))
        self.wt = nn.Parameter(torch.zeros(size=(v_dim, t_dim)))
        self.wa_v = nn.Parameter(torch.zeros(size=(t_dim, a_dim)))#update tnode
        self.wa_t = nn.Parameter(torch.zeros(size=(v_dim, a_dim)))#update vnode
        self.w1 = nn.Parameter(torch.zeros(size=(t_outdim, t_dim*2)))
        self.w2 = nn.Parameter(torch.zeros(size=(v_outdim, v_dim*2)))
        self.wa = nn.Parameter(torch.zeros(size=(a_outdim, a_dim)))
        #self.wa = nn.Parameter(torch.zeros(size=( a_dim,a_outdim)))
        self.w11 = nn.Parameter(torch.zeros(size=(t_dim, t_dim)))
        self.w12 = nn.Parameter(torch.zeros(size=(t_dim, t_dim)))
        self.act = act
        self.last_send_v = np.zeros((1, 1), dtype=np.float32)
        self.last_recv_v = np.zeros((1, 1), dtype=np.float32)
        self.send = 0
        self.send_time = 0.0

        # 进行参数初始化 xavier 初始化
        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wt)
        nn.init.xavier_uniform_(self.wa_v)
        nn.init.xavier_uniform_(self.wa_t)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.wa)
        nn.init.xavier_uniform_(self.w11)
        nn.init.xavier_uniform_(self.w12)


    def forward(self, t_info, v_info, t_embed, v_embed, a_embed, name, conn,  v_index='', v_id_send='', train=False):#adj_matrix :tensnor
        device = self.cuda
        ptr_t, a_list, v_list = t_info
        a_list = a_list.to(self.cuda)
        v_list = v_list.to(self.cuda)
        max_size = 0xffffffffffff

        ####################################################################################
        if self.tau == 0:
            if self.compress == 32:

                ## recv
                header_struct = conn.recv(4)  # recv header
                unpack_res = struct.unpack('i', header_struct)
                need_recv_size = unpack_res[0]
                v_embed_recv = b""

                b = time.time()
                while need_recv_size > 0:
                    x = conn.recv(min(max_size, need_recv_size))
                    v_embed_recv += x
                    need_recv_size -= len(x)
                e = time.time()
                self.send_time += (e - b)

                v = v_embed[v_id_send].cpu().detach().numpy()
                v_bytes = v.tobytes()
                header = struct.pack('i', len(v_bytes))
                ## send
                b = time.time()
                conn.sendall(header)
                conn.sendall(v_bytes)
                e = time.time()
                self.send_time += (e-b)

                v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.float32)).reshape(-1, self.v_dim)
                v_embed_recv = v_embed_recv[v_index]
                v_embed_recv = torch.from_numpy(np.array(v_embed_recv)).to(device)
                self.send +=1

            else:
                if self.compress == 16:
                    interval = 2/65535
                elif self.compress == 8:
                    interval = 2/255
                elif self.compress == 4:
                    interval = 2/15
                elif self.compress == 2:
                    interval = 2/3
                else:
                    interval = 2


                header_struct = conn.recv(4)
                unpack_res = struct.unpack('i', header_struct)
                need_recv_size = unpack_res[0]
                v_embed_recv = b""


                b1 = time.time()
                while need_recv_size > 0:
                    x = conn.recv(min(max_size, need_recv_size))
                    v_embed_recv += x
                    need_recv_size -= len(x)

                e1 = time.time()
                self.send_time += (e1 - b1)
                self.send +=1

                v = v_embed[v_id_send].cpu().detach().numpy()
                v_com = compress(v, interval)
                v_bytes = v_com.tobytes()
                header = struct.pack('i', len(v_bytes))

                b = time.time()
                # send
                conn.sendall(header)
                conn.sendall(v_bytes)
                e = time.time()
                self.send_time += (e - b)


                if self.compress == 8:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, self.v_dim)
                elif self.compress == 16:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint16)).reshape(-1, self.v_dim)
                elif self.compress == 4:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, int(self.v_dim/2))
                elif self.compress == 2:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, int(self.v_dim / 4))
                else:
                    v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, self.v_dim)
                v_embed_recv = decompress(v_embed_recv, interval)
                v_embed_recv = v_embed_recv[v_index]
                v_embed_recv = torch.from_numpy(np.array(v_embed_recv)).to(device)
                torch.set_printoptions(profile="full")
                print(v_embed_recv)


        else:
            interval = 2 / 15

            # recv
            header_struct = conn.recv(4)
            unpack_res = struct.unpack('i', header_struct)
            need_recv_size = unpack_res[0]
            v_embed_recv = b""

            if need_recv_size > 0:
                b = time.time()
                while need_recv_size > 0:
                    x = conn.recv(min(max_size, need_recv_size))
                    v_embed_recv += x
                    need_recv_size -= len(x)
                self.send_time += time.time() - b
                v_embed_recv = (np.frombuffer(v_embed_recv, dtype=np.uint8)).reshape(-1, int(self.v_dim/2))
                v_embed_recv = decompress(v_embed_recv, interval)
                v_embed_recv = v_embed_recv[v_index]
                if train:
                    self.last_recv_v = v_embed_recv.copy()
                v_embed_recv = torch.from_numpy(np.array(v_embed_recv)).to(device)
            else:
                v_embed_recv = torch.from_numpy(np.array(self.last_recv_v)).to(device)


            v = v_embed[v_id_send].cpu().detach().numpy()
            if train and self.last_send_v.shape[0] > 1:
                dist = np.linalg.norm(v - self.last_send_v)
                if dist <= self.tau:
                    #print("###################dist:", dist)
                    header = struct.pack('i', 0)
                    conn.sendall(header)
                else:
                    self.last_send_v = v.copy()
                    v_com = compress(v, interval)
                    v_bytes = v_com.tobytes()
                    header = struct.pack('i', len(v_bytes))
                    b = time.time()
                    conn.sendall(header)
                    conn.sendall(v_bytes)
                    self.send_time += time.time() - b
                    self.send += 1
            else:
                v_com = compress(v, interval)
                v_bytes = v_com.tobytes()
                header = struct.pack('i', len(v_bytes))
                b = time.time()
                conn.sendall(header)
                conn.sendall(v_bytes)
                self.send_time += time.time() - b
                if train:
                    self.last_send_v = v.copy()

        merge_v = torch.cat([v_embed[v_list], v_embed_recv], dim=0)
        del v_embed_recv
        del v_list
        a = a_embed.cpu().detach().numpy()
        need_recv_size = len(a.tobytes())
        a_embed_recv = b""
        while need_recv_size > 0:
            x = conn.recv(min(max_size, need_recv_size))
            a_embed_recv += x
            need_recv_size -= len(x)

        a_embed_recv = (np.frombuffer(a_embed_recv, dtype=np.float32)).reshape(a.shape[0], a.shape[1])
        a_embed_recv = torch.from_numpy(np.array(a_embed_recv)).to(device)
        a_embed_recv = a_embed_recv[a_list]
        conn.sendall(a.tobytes())


        # merge v,a
        merge_a = torch.cat([a_embed[a_list], a_embed_recv], dim=0)  # 2n * v_dim
        del a_list
        del a_embed_recv
        value_embed = self.wv.mm(merge_v.t())  # t_dim*v_dim  X v_dim*2n
        att_embed = self.wa_v.mm(merge_a.t())  # t_dim*a_dim  X a_dim*2n
        del merge_v
        del merge_a
        agg_embed = (att_embed * value_embed).t()    # 2n*tdim
        del att_embed
        del value_embed


        ptr_t = ptr_t.to(self.cuda)
        mat1 = segment_csr(agg_embed[:int(agg_embed.size(0) / 2)], ptr_t, reduce=name)
        mat2 = segment_csr(agg_embed[int(agg_embed.size(0) / 2):], ptr_t, reduce=name)
        out = (mat1 + mat2) / 2
        t_updatembed = self.act(self.w1.mm(torch.cat([t_embed, out], dim=1).t())).t()


        ptr_v, a_list, t_list = v_info
        t_list = t_list.to(self.cuda)
        tuple_embed = self.wt.mm(t_embed[t_list].t())
        a_list = a_list.to(self.cuda)
        att_embed2 = self.wa_t.mm(a_embed[a_list].t())###########

        agg_embed2 = att_embed2 * tuple_embed
        ptr_v = ptr_v.to(self.cuda)
        out2 = segment_csr(agg_embed2.t(), ptr_v, reduce=name)
        v_updatembed = self.act(self.w2.mm(torch.cat([v_embed, out2], dim=1).t())).t()
        a_updateembed = self.wa.mm(a_embed.t()).t()

        return t_updatembed, v_updatembed, a_updateembed  #torch.matmul(a_embed, self.wa)


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
