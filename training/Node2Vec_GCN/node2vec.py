import random
import numpy as np
import os
import pandas as pd
# from stellargraph.data import BiasedRandomWalk
# from stellargraph import StellarGraph



from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)


import networkx as nx
'''
    using the implementation: 
    https://stellargraph.readthedocs.io/en/stable/demos/node-classification/node2vec-weighted-node-classification.html
'''

#this function assign weights to undirected edges
def softmax(x):
    non_zero_index = [i for i in range(len(x)) if x[i] != 0]
    non_zero_values = [x[i] for i in non_zero_index]

    f_x = np.exp(non_zero_values) / np.sum(np.exp(non_zero_values))
    transition_row = []
    for i in range(len(x)):
        if i in non_zero_index:
            # import IPython
            # IPython.embed()
            # assert False
            transition_row.append(f_x[non_zero_index.index(i)])
        else:
            transition_row.append(0)
    # import IPython
    # IPython.embed()
    # assert False
    # y = np.exp(non_zero_values - np.max(non_zero_values))
    # f_x = y / np.sum(np.exp(non_zero_values))
    # y = np.exp(x - np.max(x))
    # f_x = y / np.sum(np.exp(x))
    return transition_row


def traverse(transition_matrix, visited, current_node):
    next_node = -1
    prob = -1
    for i in range(len(transition_matrix)):

        if transition_matrix[current_node][i] > prob and i not in visited:
            prob = transition_matrix[current_node][i]
            next_node = i
    return next_node


def biased_walk(transition_matrix, size):
    '''

    :param transition_matrix: transition matrix of the graph
    :param size: length of the walk
    :return: n number of flows
    '''
    #no_of_walks = 198
    #no_of_walks = 86
    no_of_walks = len(transition_matrix)
    #print("biased walk")
    starting_point_selected= []
    flows = []
    index = 0
    padding_val = len(transition_matrix)
    while index < no_of_walks:
        #print(index)
        visited = []
        flow = []
        current_node = index % len(transition_matrix)
        visited.append(current_node)
        flow.append(int(current_node))
        #flow.append(str(current_node))
        budget = size -1
        while budget > 0:
            #check the case for which we have to go back
            current_node = traverse(transition_matrix, visited, current_node)

            #if no further node exist in the walk we pad the reamining value with a padding number
            if current_node == -1:
                for pad in range(budget):
                    flow.append(padding_val)
                budget = 0
                break
            visited.append(current_node)
            flow.append(int(current_node))
            #flow.append(str(current_node))
            budget -=1
        flows.append(flow)
        index += 1
    # import IPython
    # IPython.embed()
    # assert False
    return flows

# def biased_walk(transition_matrix, size):
#     '''
#
#     :param transition_matrix: transition matrix of the graph
#     :param size: length of the walk
#     :return: n number of flows
#     '''
#     starting_point_selected= []
#     flows = []
#     index = 0
#     padding_val = len(transition_matrix)
#     while index < len(transition_matrix):
#         visited = []
#         flow = []
#         current_node = index
#         visited.append(current_node)
#         flow.append(int(current_node))
#         #flow.append(str(current_node))
#         budget = size -1
#         while budget > 0:
#             #check the case for which we have to go back
#             current_node = traverse(transition_matrix, visited, current_node)
#
#             #if no further node exist in the walk we pad the reamining value with a padding number
#             if current_node == -1:
#                 for pad in range(budget):
#                     flow.append(padding_val)
#                 budget = 0
#                 break
#             visited.append(current_node)
#             flow.append(int(current_node))
#             #flow.append(str(current_node))
#             budget -=1
#         flows.append(flow)
#         index += 1
#     return flows



def random_walks(wt_matrix, size):
    #first we have to generate a transiton matrix and then perform random walks on it
    #generating transition matrix
    transition_matrix = []
    for row in wt_matrix:
        row_transition = softmax(row)
        transition_matrix.append(row_transition)
    flows = biased_walk(transition_matrix, size)
    return flows

def fetch_dataset(dataset_path, walk_length):
    dataset_dirs =  dataset_dir = [f for f in os.listdir(dataset_path) if os.path.isdir(dataset_path + f) and "DS" not in f]
    dataset = {}
    dataset_flow_matrices = []
    flows_dataset = []
    features = []
    adjacency_matrices = []
    Y = []
    total_nodes = 0
    for dir_num in dataset_dirs:

        flow_matrix_path = "{}{}/node_graph_flow.npy".format(dataset_path, dir_num)
        adj_matrix_path = "{}{}/node_graph.adjlist".format(dataset_path, dir_num)
        flow_matrix = np.load(flow_matrix_path)
        dataset_flow_matrices.append(flow_matrix)
        output_path = "{}{}/output.npy".format(dataset_path, dir_num)
        y = np.load(output_path)
        Y.append(y)
        flow = random_walks(flow_matrix, walk_length)
        flows_dataset.append(flow)
        G = nx.read_adjlist(adj_matrix_path)
        adj = nx.adjacency_matrix(G).toarray()
        adjacency_matrices.append(adj)

        #weighted_model_flow = Word2Vec( flow, vector_size=64, window=5, min_count=0, sg=1, workers=1)
        #features.append(weighted_model_flow)
    return flows_dataset, Y, adjacency_matrices


def get_features():
    #dataset_path = "../../data/dataset/"
    dataset_path = "../../data/dataset_100_nodes_same_dominant_path/"
    #dataset_path = "../../data/dataset_50_and_different_dominant_path/"
    size_of_walk = 7
    features, Y, adjacency_matrices = fetch_dataset(dataset_path, size_of_walk)
    # import IPython
    # IPython.embed()
    # assert False
    return features, Y, adjacency_matrices



