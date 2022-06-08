import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np
import torch
import numpy

class NodeEncoder(nn.Module):
    def __init__(self, node_size, embedding_dim, rnn_dim):
        super(NodeEncoder, self).__init__()
        self.node_size = node_size  # Number of unique nodes
        self.embedding = nn.Embedding(self.node_size, embedding_dim)
        self.hidden_size = rnn_dim
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_size, batch_first=True)


    def forward(self, flows):
        """
        flows: (number_nodes, flow_length)
        """
        # print("flows")
        # import IPython
        # IPython.embed()

        nodes_embedding = self.embedding(flows)
        nodes_enc_out, nodes_enc_hidden = self.gru(nodes_embedding)
        return nodes_embedding, nodes_enc_out, nodes_enc_hidden


class GCN(nn.Module):
    def __init__(self,node_embedd, nfeat, nhid1, nhid2, nclass, dropout, adj, node_size):
        super(GCN, self).__init__()

        self.node_embedd = node_embedd
        self.gc1 = GraphConvolution(896, nhid1)
        #self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1,nhid2)
        self.gc3 = GraphConvolution(nhid2, nclass)
        self.adj = torch.randn(nclass, nclass, dtype=torch.float32, requires_grad=True)
        self.out = torch.randn(nclass,1, dtype=torch.float32, requires_grad=True)
        self.dense = nn.Linear(node_size, 1)
        #self.dense = nn.Linear(nclass, 1)
        self.dropout = dropout

    def forward(self, X, adj):
        out = []
        softmax = torch.nn.Softmax(dim=1)
        for index in range(len(X)):
            # import IPython
            # IPython.embed()
            # assert False
            # I think we pass batch.. lets see
            x, nodes_enc_out, nodes_enc_hidden = self.node_embedd(X[index])
            '''
                only for ablation study we take the embedding
            '''
            # import IPython
            # IPython.embed()
            # assert False
            x = torch.reshape(x, (50, 7 * 128))
            feature = x

            #feature = nodes_enc_hidden.squeeze(0)

            x = F.relu(self.gc1(feature, adj[index]))
            #x = F.relu(self.gc1(feature, self.adj))

            x = F.relu(self.gc2(x, adj[index]))
            #x = F.relu(self.gc2(x, self.adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc3(x, adj[index])
            #x = self.gc3(x, self.adj)

            # import IPython
            # IPython.embed()
            # assert False
            x = torch.transpose(x, 0, 1)
            x = self.dense(x)
            #x = self.out(x)


            # x = torch.mm(x, self.out)

            x = x.squeeze()
            # sig = nn.Sigmoid()
            # x = sig(x)

            #soft = nn.Softmax(dim=1)
            #x = soft(x)
            #x = torch.tensor([list(i).index(max(list(i))) for i in x], requires_grad=True, dtype=torch.float)
            # import IPython
            # IPython.embed()
            # assert False
            out.append(x)
            # softmax = nn.LogSoftmax(dim=1)
            # soft_out = softmax(x)
            # soft_out = softmax(x)
            # pred = []
            # for edge in soft_out:
            #     if edge[0] > edge[1]:
            #         pred.append(0)
            #     else:
            #         pred.append(1)
            # out.append(np.array(pred))
        # import IPython
        # IPython.embed()
        # assert False
        # import IPython
        # IPython.embed()
        # assert False
        # import IPython
        # IPython.embed()
        # assert False
        out = torch.stack(out)
        # import IPython
        # IPython.embed()
        # assert False
        out = torch.transpose(out, 0, 1)
        return out