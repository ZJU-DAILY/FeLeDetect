import csv
import torch.nn as nn
import torch.optim
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data_loader import *
import itertools
from model import Model
from utils import *
from torch.autograd import Variable
import random
import argparse
from collections import defaultdict
import time
from tensorboardX import SummaryWriter


def parse_args():
    args = argparse.ArgumentParser()

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

    args.add_argument('-dataset', default='flights',
                      help='dataset used')
    args.add_argument('-agg_func', default='mean',
                      help='agg_func in aggregation step')
    args.add_argument('-pretrained', type=bool,
                      default=False, help="use pretrained initial embeddings")

    args.add_argument('-batch_size', type=int, default=16)
    args.add_argument('-epochs', type=int, default=300)
    args.add_argument('-num_workers', type=int, default=1,
                      help='Number of processes to construct batches')
    args.add_argument('-lr', type=float, default=0.00001,
                      help='Starting Learning Rate')
    args.add_argument('-weight_d', type=float, default=1e-8,
                      help='Regularization for Optimizer')
    args.add_argument("-outfolder", "--output_folder",
                      default="./datasets/centra_datasets/movies/checkpoints/out/",
                      help="Folder name to save the models.")
    args.add_argument('-patience', type=int,
                      default=300, help="early stopping patience")
    args.add_argument('-augment', type=bool,
                      default=False, help="augment or not")
    args.add_argument('-cuda',
                      default=1, help="cuda used")
    args.add_argument('-whole', type=bool,
                      default=False)
    args.add_argument('-partial', type=bool,
                      default=False)
    args.add_argument('-version',
                      default='1')
    args = args.parse_args()
    return args


def load_data(args):
    with open('./datasets/centra_datasets/{}/attribute.csv'.format(args.dataset), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        attribute_set = [row for row in reader]
    with open('./datasets/centra_datasets/{}/train.csv'.format(args.dataset), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        train_set = [row for row in reader]
    with open('./datasets/centra_datasets/{}/valid.csv'.format(args.dataset), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        valid_set = [row for row in reader]
    with open('./datasets/centra_datasets/{}/unsample_data.csv'.format(args.dataset), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        unsample_data = [row for row in reader]

    with open('./datasets/centra_datasets/{}/sample_label.csv'.format(args.dataset), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        sample_labels = [row for row in reader]
    with open('./datasets/centra_datasets/{}/unsample_label.csv'.format(args.dataset), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        unsample_labels = [row for row in reader]
    corpus = train_set + valid_set + unsample_data
    corpus = np.array(corpus)

    train_set = np.array(train_set)
    valid_set = np.array(valid_set)
    attribute_set = np.array(attribute_set)
    print("train_set_shape", train_set.shape)
    print("valid_set_shape", valid_set.shape)
    print("corpus.shape", corpus.shape)


    train_num = train_set.shape[0] * train_set.shape[1]
    valid_num = valid_set.shape[0] * valid_set.shape[1]


    v2id = {}
    with open("./datasets/centra_datasets/{}/v2id.csv".format(args.dataset), 'r', encoding='UTF-8') as p:
        for line in p:
            x = line.rsplit(",", 1)
            k = x[0].replace('"', '')
            v2id[k] = int(x[1])
    a2id = attribute2id(attribute_set)
    corpus_triples = get_triples(corpus, v2id)

    triples = defaultdict(list)
    labels = defaultdict(list)
    triples['train'] = corpus_triples[:train_num]
    triples['valid'] = corpus_triples[train_num:train_num + valid_num]
    triples['test'] = corpus_triples
    triples['test'] = corpus_triples#[-391456:]
    train_t_info, train_v_info = get_index(triples['train'],  corpus.shape[0], len(v2id))
    valid_t_info, valid_v_info = get_index(triples['valid'], corpus.shape[0], len(v2id))
    test_t_info, test_v_info = get_index(triples['test'], corpus.shape[0], len(v2id))

    labels['test'] = list(itertools.chain(*(sample_labels + unsample_labels)))#[-391456:]
    test_arr = np.array(sample_labels + unsample_labels)#[-48932:]
    labeld_triples = defaultdict(list)
    auto_labels = defaultdict(list)
    with open('./datasets/centra_datasets/{}/train_has_label.csv'.format(args.dataset), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        f = [row for row in reader]
    for r in f:
        i, j, l = "".join(r).split()
        i = int(i)
        j = int(j)
        l = int(l)
        labeld_triples['train'].append((i, j, v2id[corpus[i][j]]))
        auto_labels['train'].append(l)

    with open('./datasets/centra_datasets/{}/valid_has_label.csv'.format(args.dataset), 'r', encoding='UTF-8') as p:
        reader = csv.reader(p)
        f = [row for row in reader]
    for r in f:
        i, j, l = "".join(r).split()
        i = int(i)
        j = int(j)
        l = int(l)
        labeld_triples['valid'].append((i, j, v2id[corpus[i][j]]))
        auto_labels['valid'].append(l)

    if args.partial:
        dex_1 = [0]
        dex_2 = [1]
        dex_3 = [2]
        dex_4 = [3]
        test_1 = test_arr[:, dex_1].tolist()
        test_2 = test_arr[:, dex_2].tolist()
        test_3 = test_arr[:, dex_3].tolist()
        test_4 = test_arr[:, dex_4].tolist()
        labels['test_1'] = list(itertools.chain(*test_1))
        labels['test_2'] = list(itertools.chain(*test_2))
        labels['test_3'] = list(itertools.chain(*test_3))
        labels['test_4'] = list(itertools.chain(*test_4))
    if args.whole:
        dex1 = [0, 1, 2, 3]
        dex2 = [4, 5, 6, 7]
        dex1_1 = [0]
        dex1_2 = [1]
        dex1_3 = [2]
        dex1_4 = [3]
        dex2_1 = [4]
        dex2_2 = [5]
        dex2_3 = [6]
        dex2_4 = [7]
        test_1 = test_arr[:, dex1].tolist()
        test_2 = test_arr[:, dex2].tolist()
        test_2_1 = test_arr[:, dex2_1].tolist()
        test_2_2 = test_arr[:, dex2_2].tolist()
        test_2_3 = test_arr[:, dex2_3].tolist()
        test_2_4 = test_arr[:, dex2_4].tolist()
        labels['test1'] = list(itertools.chain(*test_1))
        labels['test2'] = list(itertools.chain(*test_2))
        labels['test2_1'] = list(itertools.chain(*test_2_1))
        labels['test2_2'] = list(itertools.chain(*test_2_2))
        labels['test2_3'] = list(itertools.chain(*test_2_3))
        labels['test2_4'] = list(itertools.chain(*test_2_4))

        test_1_1 = test_arr[:, dex1_1].tolist()
        test_1_2 = test_arr[:, dex1_2].tolist()
        test_1_3 = test_arr[:, dex1_3].tolist()
        test_1_4 = test_arr[:, dex1_4].tolist()
        labels['test1_1'] = list(itertools.chain(*test_1_1))
        labels['test1_2'] = list(itertools.chain(*test_1_2))
        labels['test1_3'] = list(itertools.chain(*test_1_3))
        labels['test1_4'] = list(itertools.chain(*test_1_4))

    print("args.pre", args.pretrained)
    if args.pretrained:
        value_embddings, att_embeddings = init_embeddings \
            ("./datasets/centra_datasets/{}/v2vec_max.txt".format(args.dataset),
             "./datasets/centra_datasets/{}/a2vec_max.txt".format(args.dataset))
    else:

        value_embddings = np.random.randn(
            len(v2id), args.value_init_dim)
        att_embeddings = np.random.randn(
            attribute_set.shape[1], args.att_init_dim)
        print("Random Initialize.")
    np.random.seed(2021)
    tuple_embeddings = np.random.randn(
        corpus.shape[0], args.tuple_init_dim)
    return triples, labels, torch.FloatTensor(tuple_embeddings).to(device), torch.FloatTensor(value_embddings).to(
           device), torch.FloatTensor(att_embeddings).to(device), labeld_triples, auto_labels, \
           (train_t_info, train_v_info, valid_t_info, valid_v_info, test_t_info, test_v_info)


def init_embeddings(value_file, att_file):
    value_embddings, att_embeddings, tuple_embeddings = [], [], []

    with open(value_file) as f:
        for line in f:
            value_embddings.append([float(val) for val in line.strip().split()])

    with open(att_file) as f:
        for line in f:
            att_embeddings.append([float(val) for val in line.strip().split()])
    print("Initialize using bert model.")

    return np.array(value_embddings, dtype=np.float64), np.array(att_embeddings, dtype=np.float64)



def get_loss(result, label):
    return nn.CrossEntropyLoss()(result, label.squeeze())


def save_model(model, folder_name, version):
    torch.save(model.state_dict(),
               (folder_name + version + "trained.pth"))

def train(args):

    data_iter = {
        'train': DataLoader(
            TrainDataset(np.int64(labeld_triples['train']), np.int64(auto_labels['train']).reshape(-1, 1)),
            # TrainDataset(np.int64(triples['train']), np.int64(labels['train']).reshape(-1, 1)),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last= True,
            num_workers=max(0, args.num_workers),

        ),
        'valid': DataLoader(
            ValidDataset(np.int64(labeld_triples['valid']), np.int64(auto_labels['valid']).reshape(-1, 1)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0, args.num_workers),
        ),
        'test': DataLoader(
            TestDataset(np.int64(triples['test']), np.int64(labels['test']).reshape(-1, 1)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(0, args.num_workers),
        )
    }

    writer = SummaryWriter("./data/log/centra/{}".format(args.dataset))

    train_t_info, train_v_info, valid_t_info, valid_v_info, _, _ = info
    model = Model(t_init_embed, v_init_embed, a_init_embed, args, device)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)
    epoch_losses = []
    counter = 0
    best_score = None
    best_f1 = -1
    v_loss = 0

    for epoch in range(args.epochs):
        print("\nepoch-> ", epoch)
        model.train()  # getting in training mode
        epoch_loss = []

        # train
        for step, batch in enumerate(data_iter['train']):
            start_time_iter = time.time()
            optimizer.zero_grad()
            # forward
            result = model(train_t_info, train_v_info, args.agg_func, batch[0])
            # comput loss
            loss = get_loss(result, Variable(torch.LongTensor(batch[1])).to(device))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
            end_time_iter = time.time()
            # print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(step, end_time_iter - start_time_iter, loss.data.item()))
        # scheduler.step()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print("Epoch {} , average loss in train: {}".format(epoch, avg_loss))
        train_loss.append(avg_loss)
        epoch_losses.append(avg_loss)
        writer.add_scalar('average training loss--epoch', avg_loss, epoch)

        if epoch:
            model.eval()
            with torch.no_grad():
                result = model(valid_t_info, valid_v_info, args.agg_func, np.int64(labeld_triples['valid']))
                v_loss = get_loss(result, torch.LongTensor(np.int64(auto_labels['valid'])).to(device))
                pre = precision_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1))
                rec = recall_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1))
                f1 = f1_score(np.int64(auto_labels['valid']), result.cpu().data.numpy().argmax(axis=1))

                valid_loss.append(v_loss.data.item())
                valid_precision.append(pre)
                valid_recall.append(rec)
                print("valid loss:", v_loss.data.item())
                if f1 > best_f1:
                    print("Epoch: {}, val_F1 improved from {} to {}".format(epoch, best_f1, f1))
                    save_model(model, "./datasets/centra_datasets/{}/checkpoints/out/".format(args.dataset),args.version)
                    print("Done saving..................")
                    best_f1 = f1
                    counter = 0
                else:
                    counter += 1
                    print("Epoch: {}, val_F1 do not improve...............".format(epoch))
                    if counter >= args.patience:
                        break
                # save_model(model, epoch, "./datasets/centra_datasets/{}/checkpoints/out/".format(args.dataset))
    writer.close()


def detection(args):
    _, _, _, _, test_t_info, test_v_info = info
    print("len test:", len(triples['test']))
    test_edge_list, test_edge_type = get_graph(triples['test'])
    test_adj_matrix = (test_edge_list, test_edge_type)
    final_model = Model(t_init_embed, v_init_embed, a_init_embed, args, device)
    final_model.load_state_dict(torch.load(
        './datasets/centra_datasets/{}/checkpoints/out/{}trained.pth'.format(args.dataset,args.version)))
    final_model.to(device)
    final_model.eval()
    with torch.no_grad():
        result = final_model(test_t_info, test_v_info, args.agg_func, np.int64(triples['test']))
        print("Test P = {:.2f}".format( precision_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1))))
        print("Test R = {:.2f}".format(recall_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1))))
        print("Test F1 = {:.2f}".format(f1_score(np.int64(labels['test']), result.cpu().data.numpy().argmax(axis=1))))

    if args.whole:
        result = result.cpu().data.numpy()
        re = result.reshape(2224, -1)  # test num #record
        dex1 = [0, 1, 2, 3, 4, 5, 6, 7]
        dex2 = [8, 9, 10, 11, 12, 13, 14, 15]
        result1 = (re[:, dex1]).reshape(-1, 2)
        result2 = (re[:, dex2]).reshape(-1, 2)
        label = np.int64(labels['test1'])
        err_num = sum(label)
        print("[1]err_num", err_num)
        p = 0.0
        tp = 0.0
        detect = result1.argmax(axis=1)
        for i in range(detect.shape[0]):
            if int(detect[i]) == 1:
                p += 1
                if int(label[i]) == 1:
                    tp += 1
        print("[1]p", p)
        print("[1]tp", tp)

        label = np.int64(labels['test2'])
        err_num = sum(label)
        print("[2]err_num", err_num)
        p = 0.0
        tp = 0.0
        detect = result2.argmax(axis=1)
        for i in range(detect.shape[0]):
            if int(detect[i]) == 1:
                p += 1
                if int(label[i]) == 1:
                    tp += 1
        print("[2]p", p)
        print("[2]tp", tp)
        p_1 = precision_score(np.int64(labels['test1']), result1.argmax(axis=1))
        p_1 = float(('%.2f' % p_1))
        r_1 = recall_score(np.int64(labels['test1']), result1.argmax(axis=1))
        r_1 = float(('%.2f' % r_1))
        f_1 = 2 * p_1 * r_1 / (p_1 + r_1)
        print("P1:Test P: ", p_1)
        print("P1:Test R: ", r_1)
        print("P1:Test F1: ", ('%.2f' % f_1))
        p_2 = precision_score(np.int64(labels['test2']), result2.argmax(axis=1))
        p_2 = float(('%.2f' % p_2))
        r_2 = recall_score(np.int64(labels['test2']), result2.argmax(axis=1))
        r_2 = float(('%.2f' % r_2))
        f_2 = 2 * p_2 * r_2 / (p_2 + r_2)
        print("P2:Test P: ", p_2)
        print("P2:Test R: ", r_2)
        print("P2:Test F1: ", ('%.2f' % f_2))

if __name__ == '__main__':

    args = parse_args()
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print("device", device)
    set_rand_seed(2021)
    train_loss = []
    valid_loss = []
    valid_precision = []
    valid_recall = []
    triples, labels, t_init_embed, v_init_embed, a_init_embed, labeld_triples, auto_labels, info = load_data(args)
    b = time.time()
    train(args)
    print("train time", time.time() - b)
    b2 = time.time()
    detection(args)
    print("test time", time.time() - b2)
