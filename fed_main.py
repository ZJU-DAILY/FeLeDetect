import csv
import os
import sys
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from data_loader import *
import itertools
from fed_model import *
from torch.autograd import Variable
import random
import argparse
from collections import defaultdict
import time
import socket
import torch.multiprocessing as mp
from utils import *


def parse_args():
    args = argparse.ArgumentParser()
    # model args
    args.add_argument('-value_init_dim', type=int, default=768,
                      help='value node initial dimension')
    args.add_argument('-att_init_dim', type=int, default=768,
                      help='att edge initial dimension')
    args.add_argument('-tuple_init_dim', type=int, default=10,
                      help='tuple node initial dimension')

    args.add_argument('-v_outdim1', type=int, default=768,
                      help='output dimension of value node after layer1')
    args.add_argument('-a_outdim1', type=int, default=768,
                      help='output dimension of attribute node after layer1')
    args.add_argument('-t_outdim1', type=int, default=300,
                      help='output dimension of tuple node after layer1')

    args.add_argument('-v_outdim2', type=int, default=50,
                      help='output dimension of value node after layer1')
    args.add_argument('-a_outdim2', type=int, default=50,
                      help='output dimension of attribute node after layer1')
    args.add_argument('-t_outdim2', type=int, default=50,
                      help='output dimension of tuple node after layer1')

    args.add_argument('-hid_dim', type=int, default=150,
                      help='hidden dimension in linear layer')
    args.add_argument('-dropout', type=float,
                      default=0.3, help="Dropout probability for linear layer")

    args.add_argument('-dataset', default='DA4',
                      help='dataset used')
    args.add_argument('-dataset2', default='DA4',
                      help='dataset training used')

    args.add_argument('-agg_func', default='mean',
                      help='agg_func in aggregation step')
    args.add_argument('-pretrained', type=bool,
                      default=False, help="use pretrained embeddings")
    # train args
    args.add_argument('-batch_size', type=int, default=16)
    args.add_argument('-epochs', type=int, default=300)
    args.add_argument('-num_workers', type=int, default=1,
                      help='Number of processes to construct batches')
    args.add_argument('-lr', type=float, default=0.01,
                      help='Starting Learning Rate')
    args.add_argument('-weight_d', type=float, default=0.0,
                      help='Regularization for Optimizer')
    args.add_argument("-outfolder1", "--output_folder1",
                      default="./datasets/distri_datasets/beers/checkpoints/out1/",
                      help="Folder name to save the models1.")
    args.add_argument("-outfolder2", "--output_folder2",
                      default="./datasets/distri_datasets/beers/checkpoints/out2/",
                      help="Folder name to save the models2.")
    args.add_argument('-patience', type=int,
                      default=10, help="early stopping patience")
    args.add_argument('-size_ratio', type=float, default=5 / 6,  # 1,#9/10,
                      help='data size_ratio of p2 and p1')
    args.add_argument('-round', type=int, default=1,
                      help='raha round')
    args.add_argument('-version', default='')
    args.add_argument('-port', type=int, default=8800)
    args.add_argument('-cuda', default=1)
    args.add_argument('-cuda1', default=0)
    args.add_argument('-cuda2', default=1)
    args.add_argument('-comp',type=int, default=4)
    args.add_argument('-tau', type=float, default=1.5)
    args = args.parse_args()
    return args


def load_data(args, party):
    with open('./datasets/distri_datasets/{}/{}/attribute.csv'.format(args.dataset, party), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        attribute_set = [row for row in reader]
    with open('./datasets/distri_datasets/{}/{}/train.csv'.format(args.dataset, party), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        train_set = [row for row in reader]
    with open('./datasets/distri_datasets/{}/{}/valid.csv'.format(args.dataset, party), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        valid_set = [row for row in reader]
    with open('./datasets/distri_datasets/{}/{}/unsample_data.csv'.format(args.dataset, party), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        unsample_data = [row for row in reader]

    with open('./datasets/distri_datasets/{}/{}/unsample_label.csv'.format(args.dataset, party), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        unsample_labels = [row for row in reader]
    with open('./datasets/distri_datasets/{}/{}/sample_label.csv'.format(args.dataset, party), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        sample_labels = [row for row in reader]

    corpus = train_set + valid_set + unsample_data
    corpus = np.array(corpus)
    train_set = np.array(train_set)
    valid_set = np.array(valid_set)

    attribute_set = np.array(attribute_set)

    train_num = train_set.shape[0] * train_set.shape[1]
    valid_num = valid_set.shape[0] * valid_set.shape[1]

    v2id = {}
    with open("./datasets/distri_datasets/{}/{}/v2id.csv".format(args.dataset, party), 'r', encoding='UTF-8') as p:
        for line in p:
            x = line.rsplit(",", 1)
            k = x[0].replace('"', '')
            v2id[k] = int(x[1])

    a2id = attribute2id(attribute_set)
    corpus_triples = get_triples(corpus, v2id)
    triples = defaultdict(list)
    labels = defaultdict(list)

    labeld_triples = defaultdict(list)
    auto_labels = defaultdict(list)

    triples['train'] = corpus_triples[:train_num]

    triples['valid'] = corpus_triples[train_num:train_num + valid_num]

    triples['test'] = corpus_triples
    train_t_info, train_v_info = get_index(triples['train'], corpus.shape[0], len(v2id))
    valid_t_info, valid_v_info = get_index(triples['valid'], corpus.shape[0], len(v2id))
    test_t_info, test_v_info = get_index(triples['test'], corpus.shape[0], len(v2id))

    labels['test'] = list(itertools.chain(*(sample_labels+unsample_labels)))
    labels['mini_test'] = labels['test']
    test_arr = np.array(sample_labels + unsample_labels)


    with open('./datasets/distri_datasets/{}/{}/train_has_label.csv'.format(args.dataset, party),
              'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        f = [row for row in reader]

    for r in f:
        i, j, l = "".join(r).split()
        i = int(i)
        j = int(j)
        l = int(l)
        labeld_triples['train'].append((i, j, v2id[corpus[i][j]]))
        labeld_triples['train'] = labeld_triples['train']
        auto_labels['train'].append(l)
        auto_labels['train'] = auto_labels['train']

    with open('./datasets/distri_datasets/{}/{}/valid_has_label.csv'.format(args.dataset, party), 'r',
              encoding='UTF-8') as p:
        reader = csv.reader(p)
        f = [row for row in reader]
    for r in f:
        i, j, l = "".join(r).split()
        i = int(i)
        j = int(j)
        l = int(l)
        labeld_triples['valid'].append((i, j, v2id[corpus[i][j]]))
        labeld_triples['valid'] =labeld_triples['valid']#[:400*4]
        auto_labels['valid'].append(l)
        auto_labels['valid'] = auto_labels['valid']#[:400*4]


    if args.pretrained:
        value_embddings, att_embeddings = init_embeddings(
            "./datasets/distri_datasets/{}/{}/v2vec_max.txt".format(args.dataset, party),
            "./datasets/distri_datasets/{}/{}/a2vec_max.txt".format(args.dataset, party))
    else:
        value_embddings = np.random.randn(len(v2id), args.value_init_dim)
        att_embeddings = np.random.randn(len(a2id), args.att_init_dim)
    np.random.seed(2021)
    tuple_embeddings = np.random.randn(corpus.shape[0], args.tuple_init_dim)

    return triples, labels, torch.FloatTensor(tuple_embeddings), torch.FloatTensor(value_embddings), \
           torch.FloatTensor(att_embeddings), labeld_triples, auto_labels,\
           (train_t_info, train_v_info,valid_t_info, valid_v_info, test_t_info, test_v_info)


def dataloder(party, triples_train, train_label):
    if party == 1:
        datas = DataLoader(
            # TrainDataset(np.int64(triples['train']), np.int64(labels['train']).reshape(-1, 1)),
            TrainDataset(np.int64(triples_train), np.int64(train_label).reshape(-1, 1)),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=1,
            # drop_last=True
        )
    else:
        l = len(triples_train)
        datas = DataLoader(
            TrainDataset(np.int64(triples_train), np.int64(train_label).reshape(-1, 1)),
            batch_size = int(args.batch_size),
            # batch_size = 58,
            shuffle=True,
            num_workers=1,
            # drop_last=True
        )
    return datas



def get_loss(result, label):
    return nn.CrossEntropyLoss()(result, label.squeeze())



def party1_train(args, triples, labels, labeld_triples, auto_labels, t_init_embed, v_init_embed, a_init_embed,
                 data_iter, info):
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    edge_list, edge_type = get_graph(triples['train'])
    adj_matrix = (edge_list, edge_type)
    valid_edge_list, valid_edge_type = get_graph(triples['valid'])
    valid_adj_matrix = (valid_edge_list, valid_edge_type)

    train_t_info, train_v_info, valid_t_info, valid_v_info, _, _ = info
    model = Fed_Model1(t_init_embed, v_init_embed, a_init_embed, args, device, args.comp, args.tau)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # wal-amazon,sgd,0.01
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1)
    train_losses1 = []
    valid_losses1 = []
    train_pre1 = []
    train_rec1 = []
    valid_pre1 = []
    valid_rec1 = []

    host = "localhost"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, args.port))
    s.listen(5)
    print("listening...")
    conn, addr = s.accept()  #
    print('[+] Connected with', addr)  # connect
    conn.settimeout(10.0)
    counter = 0
    best_score = None
    best_f1 = -1

    # train_graph, exchage ha hv
    edge_list = adj_matrix[0]
    edge_type = adj_matrix[1]
    a = a_init_embed[edge_type]
    v = v_init_embed[edge_list[1]]
    a_send = a.cpu().detach().numpy()
    v_send = v.cpu().detach().numpy()

    # send a
    conn.sendall(a_send.tobytes())
    # recv a
    max_size = 0xffffffffffff
    need_recv_size = len(a_send.tobytes())
    a_recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        a_recv += x
        need_recv_size -= len(x)
    a_recv = (np.frombuffer(a_recv, dtype=np.float32)).reshape(a_send.shape[0], a_send.shape[1])
    a_recv = torch.from_numpy(np.array(a_recv))
    # send v
    conn.sendall(v_send.tobytes())
    # recv v
    max_size = 0xffffffffffff
    need_recv_size = len(v_send.tobytes())
    v_recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        v_recv += x
        need_recv_size -= len(x)
    v_recv = (np.frombuffer(v_recv, dtype=np.float32)).reshape(v_send.shape[0], v_send.shape[1])
    v_recv = torch.from_numpy(np.array(v_recv))#.to(device)

    # valid_graph, exchage ha hv
    edge_list2 = valid_adj_matrix[0]  # edg_list = LongTensor:[[tuples],[values]]
    edge_type2 = valid_adj_matrix[1]  # edg_type = LongTensor:[rid, rid, ...]
    a_valid = a_init_embed[edge_type2]
    v_valid = v_init_embed[edge_list2[1]]
    a_valid_send = a_valid.cpu().detach().numpy()
    v_valid_send = v_valid.cpu().detach().numpy()


    # send a
    conn.sendall(a_valid_send.tobytes())
    # recv a
    max_size = 0xffffffffffff
    need_recv_size = len(a_valid_send.tobytes())
    a_valid_recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        a_valid_recv += x
        need_recv_size -= len(x)
    a_valid_recv = (np.frombuffer(a_valid_recv, dtype=np.float32)).reshape(a_valid_send.shape[0], a_valid_send.shape[1])
    a_valid_recv = torch.from_numpy(np.array(a_valid_recv))#.to(device)

    # send v
    conn.sendall(v_valid_send.tobytes())
    # recv v
    max_size = 0xffffffffffff
    need_recv_size = len(v_valid_send.tobytes())
    v_valid_recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        v_valid_recv += x
        need_recv_size -= len(x)
    v_valid_recv = (np.frombuffer(v_valid_recv, dtype=np.float32)).reshape(v_valid_send.shape[0],
                                                                           v_valid_send.shape[1])
    v_valid_recv = torch.from_numpy(np.array(v_valid_recv))#.to(device)


    # deduplicate V
    v_id_set = set()
    v_list = adj_matrix[0][1].numpy().tolist()
    for vid in v_list:
        v_id_set.add(vid)
    v_id_list = list(v_id_set)
    v_id_list.sort()  # vid_list vembed[vid_list]
    v_list_dict = {}
    k = 0
    for v in v_id_list:
        v_list_dict[v] = k
        k += 1
    v_index = []  # use v_idex ge vembed
    for index in v_list:
        v_index.append(v_list_dict[index])

    v_index = np.array(v_index)
    header = struct.pack('i', len(v_index.tobytes()))
    conn.send(header)
    conn.sendall(v_index.tobytes())

    header_struct = conn.recv(4)  # 4 length
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    v_index_recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        v_index_recv += x
        need_recv_size -= len(x)
    v_index_recv = (np.frombuffer(v_index_recv, dtype=np.int)).reshape(1, -1).flatten()
    v_index_recv = torch.LongTensor(v_index_recv)
    v_id_list = torch.LongTensor(v_id_list)

    v_id_set_valid = set()
    v_list_valid = valid_adj_matrix[0][1].numpy().tolist()
    for i in v_list_valid:
        v_id_set_valid.add(i)
    v_id_list_valid = list(v_id_set_valid)
    v_id_list_valid.sort()  # vid_list send vembed[vid_list]
    v_list_dict_valid = {}
    k = 0
    for v in v_id_list_valid:
        v_list_dict_valid[v] = k
        k += 1
    v_index_valid = []  # use v_idex get vembed
    for index in v_list_valid:
        v_index_valid.append(v_list_dict_valid[index])

    v_index_valid = np.array(v_index_valid)
    header = struct.pack('i', len(v_index_valid.tobytes()))
    conn.send(header)
    conn.sendall(v_index_valid.tobytes())

    header_struct = conn.recv(4)  # 4 length
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    v_index_recv_valid = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        v_index_recv_valid += x
        need_recv_size -= len(x)
    v_index_recv_valid = (np.frombuffer(v_index_recv_valid, dtype=np.int)).reshape(1, -1).flatten()
    v_index_recv_valid = torch.LongTensor(v_index_recv_valid)
    v_id_list_valid = torch.LongTensor(v_id_list_valid)

    for epoch in range(args.epochs):
        model.train()  # getting in training mode
        epoch_loss = []
        pre_list = []
        rec_list = []
        l = 0
        r = 0
        for step, batch in enumerate(data_iter):
            optimizer.zero_grad()
            result = model(train_t_info, train_v_info, args.agg_func, batch[0], a_recv, v_recv, conn, v_index_recv, v_id_list, True)
            loss = get_loss(result, Variable(batch[1]).to("cuda:1"))#.to(device))
            pre = precision_score(batch[1], result.cpu().data.numpy().argmax(axis=1), average="binary",zero_division=0)
            rec = recall_score(batch[1], result.cpu().data.numpy().argmax(axis=1), average="binary",zero_division=0)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
            pre_list.append(pre)
            rec_list.append(rec)
            # print("p1, Epoch->{}, Iteration->{} , Iteration_loss{}".format(epoch, step, loss.data.item()))
        # scheduler.step()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_pre = sum(pre_list) / len(pre_list)
        avg_rec = sum(rec_list) / len(rec_list)
        # print("p1, Epoch {} , average loss in train: {}".format(epoch, avg_loss))
        train_losses1.append(avg_loss)
        train_pre1.append(avg_pre)
        train_rec1.append(avg_rec)
        # writer.add_scalar('average training loss--epoch', avg_loss, epoch)

        if epoch >= 0:
            model.eval()
            with torch.no_grad():
                result = model(valid_t_info, valid_v_info, args.agg_func, np.int64(labeld_triples['valid']), a_valid_recv, v_valid_recv, conn,
                               v_index_recv_valid, v_id_list_valid)
                valid_loss = get_loss(result, torch.LongTensor(np.int64(auto_labels['valid'])).to("cuda:1"))#.to(device))
                valid_pre = precision_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1), zero_division=0)
                valid_rec = recall_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1), zero_division=0)
                valid_f1 = f1_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1), zero_division=0)
                valid_losses1.append(valid_loss)
                valid_pre1.append(valid_pre)
                valid_rec1.append(valid_rec)
                print("p1, Epoch {} , average loss in valid: {}".format(epoch, valid_loss))
            if valid_f1 > best_f1:
                save_model(model, "./datasets/distri_datasets/{}/checkpoints/out1/".format(args.dataset), args.version)
                print("p1's val_f1 imporve from {} to {}".format(best_f1, valid_f1))
                best_f1 = valid_f1
                print("#####################################################################Done saving (1)...")
    # save_model(model, "./datasets/distri_datasets/{}/checkpoints/out1/".format(args.dataset),args.version)
    #writer.close()
    conn.close()
    s.close()
    print("P1 totally send_num:", model.layer2.send)
    print("P1 totally trans_time:", model.layer2.send_time)
    print("P1 totally trans_bytes:", model.layer2.exchange_cost)
    train_losses1 = np.array(train_losses1)
    np.savetxt('./icde/loss/{}_1_{}.csv'.format(args.dataset, args.comp), train_losses1, delimiter=",")
    # np.savetxt('./icde/loss/c4_{}_1_{}_tau{}.csv'.format(args.dataset, args.comp, args.tau), train_losses1, delimiter=",")


def party2_train(args, triples, labels, labeld_trples, auto_labels, t_init_embed, v_init_embed, a_init_embed,
                 data_iter, info):
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    edge_list, edge_type = get_graph(triples['train'])
    adj_matrix = (edge_list, edge_type)
    valid_edge_list, valid_edge_type = get_graph(triples['valid'])
    valid_adj_matrix = (valid_edge_list, valid_edge_type)

    train_t_info, train_v_info, valid_t_info, valid_v_info, _, _ = info
    model = Fed_Model2(t_init_embed, v_init_embed, a_init_embed, args,device, args.comp, args.tau)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8, last_epoch=-1)
    train_losses2 = []
    valid_losses2 = []
    train_pre2 = []
    train_rec2 = []
    valid_pre2 = []
    valid_rec2 = []
    model.train()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "localhost"
    try:
        s.connect((host, args.port))  # connect server
    except Exception:
        print('[!] Server not found or not open')
        sys.exit()
    counter = 0
    best_score = None
    best_f1 = -1
    ########train
    edge_list = adj_matrix[0]  # edg_list = LongTensor:[[tuples],[values]]
    edge_type = adj_matrix[1]  # edg_type = LongTensor:[rid, rid, ...]
    # edge_list = edge_list.to(device)
    # edge_type = edge_type.to(device)
    a = a_init_embed[edge_type]
    v = v_init_embed[edge_list[1]]
    a_send = a.cpu().detach().numpy()
    v_send = v.cpu().detach().numpy()

    # recv a
    max_size = 0xffffffffffff
    need_recv_size = len(a_send.tobytes())
    a_recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        a_recv += x
        need_recv_size -= len(x)
    a_recv = (np.frombuffer(a_recv, dtype=np.float32)).reshape(a_send.shape[0], a_send.shape[1])
    a_recv = torch.from_numpy(np.array(a_recv))#.to(device)
    # send a
    s.sendall(a_send.tobytes())

    # recv v
    max_size = 0xffffffffffff
    need_recv_size = len(v_send.tobytes())
    v_recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        v_recv += x
        need_recv_size -= len(x)
    v_recv = (np.frombuffer(v_recv, dtype=np.float32)).reshape(v_send.shape[0], v_send.shape[1])
    v_recv = torch.from_numpy(np.array(v_recv))#.to(device)
    # send v
    s.sendall(v_send.tobytes())

    ###########valid
    edge_list2 = valid_adj_matrix[0]  # edg_list = LongTensor:[[tuples],[values]]
    edge_type2 = valid_adj_matrix[1]  # edg_type = LongTensor:[rid, rid, ...]
    # edge_list2 = edge_list2.to(device)
    # edge_type2 = edge_type2.to(device)
    a_valid = a_init_embed[edge_type2]
    v_valid = v_init_embed[edge_list2[1]]

    a_valid_send = a_valid.cpu().detach().numpy()
    v_valid_send = v_valid.cpu().detach().numpy()

    # recv a
    max_size = 0xffffffffffff
    need_recv_size = len(a_valid_send.tobytes())
    a_valid_recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        a_valid_recv += x
        need_recv_size -= len(x)
    a_valid_recv = (np.frombuffer(a_valid_recv, dtype=np.float32)).reshape(a_valid_send.shape[0],
                                                                           a_valid_send.shape[1])
    a_valid_recv = torch.from_numpy(np.array(a_valid_recv))#.to(device)
    # send a
    s.sendall(a_valid_send.tobytes())

    # recv v
    max_size = 0xffffffffffff
    need_recv_size = len(v_valid_send.tobytes())
    v_valid_recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        v_valid_recv += x
        need_recv_size -= len(x)
    v_valid_recv = (np.frombuffer(v_valid_recv, dtype=np.float32)).reshape(v_valid_send.shape[0],
                                                                           v_valid_send.shape[1])
    v_valid_recv = torch.from_numpy(np.array(v_valid_recv))#.to(device)
    # send v
    s.sendall(v_valid_send.tobytes())

    ############# train
    v_id_set = set()
    v_list = adj_matrix[0][1].numpy().tolist()
    for i in v_list:
        v_id_set.add(i)
    v_id_list = list(v_id_set)
    v_id_list.sort()
    v_list_dict = {}
    k = 0
    for v in v_id_list:
        v_list_dict[v] = k
        k += 1
    v_index = []
    for index in v_list:
        v_index.append(v_list_dict[index])

    header_struct = s.recv(4)  # receive 4
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    v_index_recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        v_index_recv += x
        need_recv_size -= len(x)
    v_index_recv = (np.frombuffer(v_index_recv, dtype=np.int)).reshape(1, -1).flatten()

    v_index = np.array(v_index)
    header = struct.pack('i', len(v_index.tobytes()))
    s.send(header)
    s.sendall(v_index.tobytes())

    v_index_recv = torch.LongTensor(v_index_recv)
    v_id_list = torch.LongTensor(v_id_list)

    #### valid
    v_id_set_valid = set()
    v_list_valid = valid_adj_matrix[0][1].numpy().tolist()
    for i in v_list_valid:
        v_id_set_valid.add(i)
    v_id_list_valid = list(v_id_set_valid)
    v_id_list_valid.sort()  # vid_list send vembed[vid_list]
    v_list_dict_valid = {}
    k = 0
    for v in v_id_list_valid:
        v_list_dict_valid[v] = k
        k += 1
    v_index_valid = []  # v_idex return vembed
    for index in v_list_valid:
        v_index_valid.append(v_list_dict_valid[index])

    header_struct = s.recv(4)  #
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    v_index_recv_valid = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        v_index_recv_valid += x
        need_recv_size -= len(x)
    v_index_recv_valid = (np.frombuffer(v_index_recv_valid, dtype=np.int)).reshape(1, -1).flatten()

    v_index_valid = np.array(v_index_valid)
    header = struct.pack('i', len(v_index_valid.tobytes()))
    s.send(header)
    s.sendall(v_index_valid.tobytes())

    v_index_recv_valid = torch.LongTensor(v_index_recv_valid)
    v_id_list_valid = torch.LongTensor(v_id_list_valid)


    for epoch in range(args.epochs):

        #print("\np2 epoch-> ", epoch)
        model.train()  # getting in training mode
        epoch_loss = []
        pre_list = []
        rec_list = []
        l = 0
        r = 0
        for step, batch in enumerate(data_iter):
            optimizer.zero_grad()
            result = model(train_t_info, train_v_info, args.agg_func, batch[0], a_recv, v_recv, s, v_index_recv, v_id_list, True)
            loss = get_loss(result, Variable(batch[1]).to("cuda:3"))#.to(device))
            pre = precision_score(batch[1], result.cpu().data.numpy().argmax(axis=1), average="binary",zero_division=0)
            rec = recall_score(batch[1], result.cpu().data.numpy().argmax(axis=1), average="binary",zero_division=0)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
            pre_list.append(pre)
            rec_list.append(rec)
            # print("p2, Epoch->{}, Iteration->{} , Iteration_loss{}".format(epoch, step, loss.data.item()))
        # scheduler.step()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_pre = sum(pre_list) / len(pre_list)
        avg_rec = sum(rec_list) / len(rec_list)
        print("p2, Epoch {} , average loss in train: {}".format(epoch, avg_loss))
        train_losses2.append(avg_loss)
        train_pre2.append(avg_pre)
        train_rec2.append(avg_rec)
        ###########
        if epoch >= 0:
            model.eval()
            with torch.no_grad():
                result = model( valid_t_info, valid_v_info, args.agg_func, np.int64(labeld_trples['valid']), a_valid_recv, v_valid_recv, s,
                               v_index_recv_valid, v_id_list_valid)
                valid_loss = get_loss(result, torch.LongTensor(np.int64(auto_labels['valid'])).to("cuda:3"))#.to(device))
                valid_pre = precision_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1),zero_division=0)
                valid_rec = recall_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1),zero_division=0)
                valid_f1 = f1_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1),zero_division=0)
                valid_losses2.append(valid_loss)
                valid_pre2.append(valid_pre)
                valid_rec2.append(valid_rec)
                #print("p2, Epoch {} , average loss in valid: {}".format(epoch, valid_loss))
            if valid_f1 > best_f1:
                save_model(model, "./datasets/distri_datasets/{}/checkpoints/out2/".format(args.dataset), args.version)
                print("p2's val_f1 imporve from {} to {}".format(best_f1, valid_f1))
                best_f1 = valid_f1
                print("###################################################################Done saving (2)...")


    s.close()
    print("P2 totally send_num:", model.layer2.send)
    print("P2 totally trans_time", model.layer2.send_time)
    train_losses2 = np.array(train_losses2)
    np.savetxt('./icde/loss/{}_2_{}.csv'.format(args.dataset, args.comp), train_losses2, delimiter=",")
    # np.savetxt('./icde/loss/c4_{}_2_{}_tau{}.csv'.format(args.dataset, args.comp, args.tau), train_losses2, delimiter=",")

def party1_evaluate(args, triples, labels, t_init_embed, v_init_embed, a_init_embed, info):
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    _, _, _, _, test_t_info, test_v_info = info
    test_edge_list, test_edge_type = get_graph(triples['test'])
    test_adj_matrix = (test_edge_list, test_edge_type)

    final_model = Fed_Model1(t_init_embed, v_init_embed, a_init_embed, args, device, args.comp, args.tau)
    final_model.load_state_dict(torch.load(
        "./datasets/distri_datasets/{}/checkpoints/out1/{}trained.pth".format(args.dataset, args.version)))
    #final_model.to(device)
    final_model.eval()
    host = "localhost"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, args.port))
    s.listen(10)
    print("listening...")
    conn, addr = s.accept()  #
    print('[+] Connected with', addr)  #

    print("1 ready..")
    #
    edge_list = test_adj_matrix[0]  # edg_list = LongTensor:[[tuples],[values]]
    edge_type = test_adj_matrix[1]  # edg_type = LongTensor:[rid, rid, ...]
    # edge_list = edge_list.to(device)
    # edge_type = edge_type.to(device)
    a = a_init_embed[edge_type]
    v = v_init_embed[edge_list[1]]

    a_send = a.cpu().detach().numpy()
    v_send = v.cpu().detach().numpy()
    # MakePrivate(a_send, b=0.1, seed=2021)
    # MakePrivate(v_send, b=0.1, seed=2021)

    conn.sendall(a_send.tobytes())

    max_size = 0xffffffffffff
    need_recv_size = len(a_send.tobytes())
    a_recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        a_recv += x
        need_recv_size -= len(x)
    a_recv = (np.frombuffer(a_recv, dtype=np.float32)).reshape(a_send.shape[0], a_send.shape[1])
    a_recv = torch.from_numpy(np.array(a_recv))#.to(device)

    conn.sendall(v_send.tobytes())
    max_size = 0xffffffffffff
    need_recv_size = len(v_send.tobytes())
    v_recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        v_recv += x
        need_recv_size -= len(x)
    v_recv = (np.frombuffer(v_recv, dtype=np.float32)).reshape(v_send.shape[0], v_send.shape[1])
    v_recv = torch.from_numpy(np.array(v_recv))#.to(device)

    #####vembed
    v_id_set = set()
    v_list = test_adj_matrix[0][1].numpy().tolist()
    for i in v_list:
        v_id_set.add(i)
    v_id_list = list(v_id_set)
    v_id_list.sort()
    v_list_dict = {}
    k = 0
    for v in v_id_list:
        v_list_dict[v] = k
        k += 1
    v_index = []
    for index in v_list:
        v_index.append(v_list_dict[index])

    v_index = np.array(v_index)
    header = struct.pack('i', len(v_index.tobytes()))
    conn.send(header)
    conn.sendall(v_index.tobytes())

    header_struct = conn.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    v_index_recv = b""
    while need_recv_size > 0:
        x = conn.recv(min(max_size, need_recv_size))
        v_index_recv += x
        need_recv_size -= len(x)
    v_index_recv = (np.frombuffer(v_index_recv, dtype=np.int)).reshape(1, -1).flatten()
    v_index_recv = torch.LongTensor(v_index_recv)
    v_id_list = torch.LongTensor(v_id_list)


    with torch.no_grad():
        result = final_model(test_t_info, test_v_info, args.agg_func, np.int64(triples['test']), a_recv, v_recv, conn, v_index_recv,
                             v_id_list)

        print("p1,Test P:",
              precision_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1),zero_division=0))
        print("p1,Test R:",
              recall_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1),zero_division=0))
        print("p1,Test F1:", f1_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1),zero_division=0))
    conn.close()
    s.close()


def party2_evaluate(args, triples, labels, t_init_embed, v_init_embed, a_init_embed, info):
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    _, _, _, _, test_t_info, test_v_info = info
    test_edge_list, test_edge_type = get_graph(triples['test'])
    test_adj_matrix = (test_edge_list, test_edge_type)

    final_model = Fed_Model2(t_init_embed, v_init_embed, a_init_embed, args, device, args.comp, args.tau)
    final_model.load_state_dict(torch.load("./datasets/distri_datasets/{}/checkpoints/out2/{}trained.pth".format(
        args.dataset, args.version)))
    final_model.eval()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "localhost"
    try:
        s.connect((host, args.port))  #
    except Exception:
        print('[!] Server not found or not open')
        sys.exit()
    print("2 ready..")

    edge_list = test_adj_matrix[0]  # edg_list = LongTensor:[[tuples],[values]]
    edge_type = test_adj_matrix[1]  # edg_type = LongTensor:[rid, rid, ...]
    a = a_init_embed[edge_type]
    v = v_init_embed[edge_list[1]]
    a_send = a.cpu().detach().numpy()
    v_send = v.cpu().detach().numpy()

    max_size = 0xffffffffffff
    need_recv_size = len(a_send.tobytes())
    a_recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        a_recv += x
        need_recv_size -= len(x)
    a_recv = (np.frombuffer(a_recv, dtype=np.float32)).reshape(a_send.shape[0], a_send.shape[1])
    a_recv = torch.from_numpy(np.array(a_recv))#.to(device)
    s.sendall(a_send.tobytes())

    need_recv_size = len(v_send.tobytes())
    v_recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        v_recv += x
        need_recv_size -= len(x)
    v_recv = (np.frombuffer(v_recv, dtype=np.float32)).reshape(v_send.shape[0], v_send.shape[1])
    v_recv = torch.from_numpy(np.array(v_recv))#.to(device)
    s.sendall(v_send.tobytes())


    ######## vembed
    v_id_set = set()
    v_list = test_adj_matrix[0][1].numpy().tolist()
    for i in v_list:
        v_id_set.add(i)
    v_id_list = list(v_id_set)
    v_id_list.sort()  # vid_list vembed[vid_list]
    v_list_dict = {}
    k = 0
    for v in v_id_list:
        v_list_dict[v] = k
        k += 1
    v_index = []  # v_idex
    for index in v_list:
        v_index.append(v_list_dict[index])

    header_struct = s.recv(4)
    unpack_res = struct.unpack('i', header_struct)
    need_recv_size = unpack_res[0]
    v_index_recv = b""
    while need_recv_size > 0:
        x = s.recv(min(max_size, need_recv_size))
        v_index_recv += x
        need_recv_size -= len(x)
    v_index_recv = (np.frombuffer(v_index_recv, dtype=np.int)).reshape(1, -1).flatten()

    v_index = np.array(v_index)
    header = struct.pack('i', len(v_index.tobytes()))
    s.send(header)
    s.sendall(v_index.tobytes())

    v_index_recv = torch.LongTensor(v_index_recv)
    v_id_list = torch.LongTensor(v_id_list)
    with torch.no_grad():
        result = final_model(test_t_info, test_v_info, args.agg_func, np.int64(triples['test']), a_recv, v_recv, s, v_index_recv,
                             v_id_list)
        print("p2,Test P:",
              precision_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1),zero_division=0))
        print("p2,Test R:",
              recall_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1),zero_division=0))
        print("p2,Test F1:", f1_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1),zero_division=0))

    s.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    set_rand_seed(2021)
    triples1, labels1, t_init_embed1, v_init_embed1, a_init_embed1, labeld_triples1, auto_labels1, info1 = load_data(args, 1)
    set_rand_seed(2021)
    triples2, labels2, t_init_embed2, v_init_embed2, a_init_embed2, labeld_triples2, auto_labels2, info2 = load_data(args, 2)

    data_iter1 = dataloder(1, labeld_triples1['train'], auto_labels1['train'])
    data_iter2 = dataloder(2, labeld_triples2['train'], auto_labels2['train'])

    begin = time.time()
    p1 = mp.Process(target=party1_train,
                    args=(
                    args, triples1, labels1, labeld_triples1, auto_labels1, t_init_embed1, v_init_embed1, a_init_embed1,
                    data_iter1, info1,))
    p2 = mp.Process(target=party2_train,
                    args=(
                    args, triples2, labels2, labeld_triples2, auto_labels2, t_init_embed1, v_init_embed2, a_init_embed2,
                    data_iter2, info2,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end = time.time()
    print("training time:", end - begin)
    print("training process completed")
    print("test begin...")
    b = time.time()
    p3 = mp.Process(target = party1_evaluate,
                    args=(args, triples1, labels1, t_init_embed1, v_init_embed1, a_init_embed1, info1,))
    p4 = mp.Process(target = party2_evaluate,
                    args=(args, triples2, labels2, t_init_embed1, v_init_embed2, a_init_embed2, info2,))
    p3.start()
    p4.start()
    p3.join()
    p4.join()
    e = time.time()
    print("test time:", e - b)
