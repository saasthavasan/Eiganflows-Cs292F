# import pandas as pd
# import os
#
# import stellargraph as sg
# from stellargraph.mapper import FullBatchNodeGenerator
# from stellargraph.layer import GCN
#
# from tensorflow.keras import layers, optimizers, losses, metrics, Model
# from sklearn import preprocessing, model_selection
# from IPython.display import display, HTML
# import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import GCN, NodeEncoder
from dataloader import Dataset
from node2vec import get_features
from utils import calc_accuracy
# from __future__ import division
# from __future__ import print_function

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Number of hidden units.')
parser.add_argument('--rnn-dim', type=int, default=256,
                        help='Number of hidden units.')
parser.add_argument('--hidden1', type=int, default=128,
                        help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=32,
                        help='Number of hidden units.')
# parser.add_argument('--embedding-dim', type=int, default=128,
#                         help='Number of hidden units.')
# parser.add_argument('--rnn-dim', type=int, default=128,
#                         help='Number of hidden units.')
# parser.add_argument('--hidden1', type=int, default=64,
#                         help='Number of hidden units.')
# parser.add_argument('--hidden2', type=int, default=32,
#                         help='Number of hidden units.')

parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
device = "cpu"
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#features: total_filesXnum_of_nodesXsize_of_walk
#Y: size_of_classes
X, Y, adj = get_features()


#+1 for padding
total_nodes = len(adj[0])
node_encoder = NodeEncoder(total_nodes, args.embedding_dim, args.rnn_dim)
# labels = {tech: ll for tech, ll in Y.items() if len([l for l in ll if l == 1]) >= 200}
labels = {}
for graph in Y:
    for edge in range(len(graph)):
        if str(edge) not in labels:
            labels[str(edge)] = []
        if graph[edge] == 1:
            labels[str(edge)].append(1)
        else:
            labels[str(edge)].append(0)
dataset = Dataset(X, labels)
#fetching all the keys from labels{}
unique_labels = [l for l in labels]
training_set_length = int(0.8*len(dataset))
testing_set_length = int(0.2*len(dataset))

train_set, validation_set = torch.utils.data.random_split(dataset, [training_set_length, testing_set_length])

# Y = torch.tensor(Y, device=device, dtype=torch.float)
# X_train = torch.tensor(X, device=device)
adj = torch.tensor(adj, device=device)
# node_id_str_to_int = {str(i) for i in range(len(X[0]))}

#number of nodes in input will be total number of nodes + 1

#adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer

train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=32)
test_loader = torch.utils.data.DataLoader(validation_set, shuffle=True, batch_size=32)
model = GCN(node_embedd=node_encoder,
            nfeat=args.rnn_dim,
            nhid1=args.hidden1,
            nhid2=args.hidden2,
            nclass=len(labels),
            dropout=args.dropout,
            adj=adj[0], node_size=total_nodes)


optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.MultiLabelSoftMarginLoss()
#criterion = nn.MSELoss()
#criterion = torch.nn.NLLLoss()
best_validation_accuracy = 0
for epoch in range(1, args.epochs + 1):
    #Train
    output_for_each_epoch = []
    train_loss = 0
    corrects = 0
    total = 0
    train_accuracy = 0
    for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
        batch_inputs = batch_inputs.to(device)
        batch_labels = [bl.to(device, dtype=torch.float) for bl in batch_labels]
        optimizer.zero_grad()

        outputs = model(batch_inputs, adj)

        train_losses = [criterion(output, b_labels) for b_labels, output in zip(batch_labels, outputs)]

        loss = sum(train_losses) / len(train_losses)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_correct, batch_total = calc_accuracy(outputs, batch_labels)
        corrects += batch_correct
        total += batch_total

    train_accuracy = float(corrects) / total

    train_loss = (loss / (batch_idx + 1))

   #Test
    test_loss = 0
    corrects = 0
    total = 0
    validation_accuracy = 0

    for batch_idx, (batch_inputs, batch_labels) in enumerate(test_loader):
        batch_inputs = batch_inputs.to(device)
        batch_labels = [bl.to(device, dtype=torch.float) for bl in batch_labels]
        with torch.no_grad():
            outputs = model(batch_inputs, adj)
            losses = [criterion(output, b_labels) for b_labels, output in zip(batch_labels, outputs)]
            loss = sum(losses) / len(losses)
            test_loss += loss.item()
            batch_correct, batch_total = calc_accuracy(outputs, batch_labels)
            corrects += batch_correct
            total += batch_total
    validation_accuracy = float(corrects) / total
    if best_validation_accuracy <= validation_accuracy:
        best_validation_accuracy = validation_accuracy
    test_loss = (test_loss/(batch_idx + 1))

    print("Epoch {}, Train_Loss: {}, Validation_Loss : {}, Train_accuracy: {}, Validation_accuracy: {}"\
          .format(epoch, train_loss, test_loss, train_accuracy, validation_accuracy))

print("\n--------hallelujah---------\n")
import IPython
IPython.embed()
assert False

#for 100 nodes:
# best flow
#[15, 51, 93, 98, 97, 76]
