import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx.generators.random_graphs as nxrg
from itertools import *
import os
import json

def crazy_BFS(adj, src, dest, v):
    # a queue to maintain queue of vertices
    queue = []

    # boolean array visited[] which stores the
    # information whether ith vertex is reached
    # at least once in the Breadth first search
    visited = [False for i in range(v)]

    # initially all vertices are unvisited
    # so v[i] for all i is false
    # and as no path is yet constructed
    # dist[i] for all i set to infinity
    pred = [-1 for i in range(v)]
    distance = [v + 1 for i in range(v)]

    # now source is first to be visited and
    # distance from source to itself should be 0
    visited[src] = True
    distance[src] = 0
    queue.append(src)
    flag = 0
    # standard BFS algorithm... is it?
    while len(queue) != 0:
        if flag == 1:
            break
        #fetch the next node
        #print("Queue: {}".format(queue))
        cur = queue[0]
        #pop the node from queue

        val = queue.pop(0)
        #print("popped {}".format(val))
        for i in range(len(adj[cur])):
            if flag == 1:
                break
            #print(i)
            # checking if cur node has already been visited if not then we make it true
            if not visited[i] and adj[cur, i] == 1 and cur != i:
                visited[i] = True
                #print("{} added to visited".format(i))
                distance[i] = distance[cur] + 1
                pred[i] = cur
                queue.append(i)
                # we stop BFS when we find the destination
                if i == dest:
                    queue = []
                    flag = 1
                    break

    # we return empty list if destination is not found during BFS
    return pred, distance


def fetch_shortest_path(adj, source, dest, v):

    pred, distance = crazy_BFS(adj, source, dest, v)
    #print("return successful")
    if pred[dest] == -1:
        print("Path not found")
        return [], -1
    shortest_path = [dest]
    current_node = dest
    #creating the shortest path from pred
    while(1):
        if current_node == source:
            break
        current_node = pred[current_node]
        shortest_path.append(current_node)
    shortest_path = shortest_path[::-1]
    print("shortest path: {}".format(shortest_path))
    return shortest_path, distance[dest]


def adjacency_gen(nnodes, undirected = False):
    adj = np.random.randint(2, size = (nnodes, nnodes), dtype = int)
    for i in range(nnodes):
        adj[i,i] = 0
    if(undirected):
        for i in range(nnodes):
            for j in range(i):
                adj[i,j] = adj[j,i]
    return adj


def assign_weights(G, n, shortest_path):
    wt_matrix = np.zeros((n, n))

    # we assign high weights to edges that are in the shortest path
    for i in range(len(shortest_path)-1):

        wt = random.randint(50, 60)
        wt_matrix[shortest_path[i],shortest_path[i+1]] = wt
        wt_matrix[shortest_path[i+1], shortest_path[i]] = wt
        #print("{},{}, {}".format(shortest_path[i], shortest_path[i + 1], wt))

    # we assign really low weights to edges that are not in shortest path
    for i in range(n):
        for j in range(n):
            if wt_matrix[i, j] == 0 and G[i, j] == 1:
                wt = random.randint(0, 10)
                wt_matrix[i, j] = wt
                wt_matrix[j, i] = wt

    return wt_matrix

def make_double_directed(G):
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i, j] == 1:
                G[j, i] = 1
    return G


def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


def generate_adj_matrix_for_edge_graph(new_node_positions, new_node_ids):
    adj = np.zeros((len(new_node_ids), len(new_node_ids)))
    node_id_pos_dict = {  new_node_ids[node_id]: new_node_positions[node_id]for node_id in new_node_ids}
    '''
    position is stored in the form of [1, 2], [1,3]... they are basically the edge in the node graph
    '''
    edges = []
    for index1 in range(len(new_node_positions) -1):
        source = new_node_positions[index1]
        edge_node_source = new_node_positions[index1][0]
        edge_node_dest = new_node_positions[index1][1]
        index2 = index1 + 1

        # just the implementation of algorithm to generate edge
        while index2 < len(new_node_positions):
            if new_node_positions[index2][0] == edge_node_source or  new_node_positions[index2][0] == edge_node_dest:
                dest = new_node_positions[index2]
                source_id = ([node_id for node_id in node_id_pos_dict if node_id_pos_dict[node_id] == source])[0]
                dest_id = ([node_id for node_id in node_id_pos_dict if node_id_pos_dict[node_id] == dest])[0]
                if [source_id, dest_id] or [dest_id, source_id] not in edges:
                    edges.append([source_id, dest_id])

            elif new_node_positions[index2][1] == edge_node_source or new_node_positions[index2][1] == edge_node_dest:
                dest = new_node_positions[index2]
                source_id = ([node_id for node_id in node_id_pos_dict if node_id_pos_dict[node_id] == source])[0]
                dest_id = ([node_id for node_id in node_id_pos_dict if node_id_pos_dict[node_id] == dest])[0]
                if [source_id, dest_id] or [dest_id, source_id] not in edges:
                    edges.append([source_id, dest_id])

            index2 += 1

    #filling adjacency matrix
    for edge in edges:
        adj[edge[0], edge[1]] = 1
        adj[edge[1], edge[0]] = 1

    return adj, edges



def write_data(counter, g_object, edge_graph_adj, node_graph_flow, edge_graph_features, node_id_pos_dict):
    '''

    :param counter: it just stores the iteration name(for writing the data)
    :param g_object: it is the edge adj matrix of the actual node graph
    :param edge_graph_adj: it is the edge adj matrix of the converted edge graph
    :param node_graph_flow: it is the weight matrix/flow matrix of the node graph
    :param node_graph_to_edge_graph_conv: it provides information on what each edge represents (can be used to covert)
            the edge graph back to node graph
    :param edge_graph_features: it consist of a list of edges and node features of the edge graph
    :param node_id_pos_dict: dictionary containing information of edgegraph_node_id: nodegraph_edge_src_dst
    :return: nothing
    '''

    output_dir = "{}/{}/".format(os.getcwd(), "dataset")
    #checking if dataset directory exist, if not create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print("created directory: {}".format(output_dir))

    output_dir += "{}/".format(counter)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print("created directory: {}".format(output_dir))

    # writing node_graph_adjacency_list
    nx.write_adjlist(g_object, "{}{}".format(output_dir, "node_graph.adjlist"))

    # writing edge graph adjacency list
    edge_graph_obj =  nx.from_numpy_matrix(edge_graph_adj)

    edge_graph_obj.add_edges_from(edge_graph_features["edges"])
    nx.write_adjlist(edge_graph_obj, "{}{}".format(output_dir, "node_graph.adjlist"))

    # with open("{}node_graph_flow.json".format(output_dir), "w") as my_file:
    #     import IPython
    #     IPython.embed()
    #     assert False
    #     json.dump(node_flow, my_file)
    np.save("{}node_graph_flow".format(output_dir), node_graph_flow)

    with open("{}node_graph_to_edge_graph_conv.json".format(output_dir), "w") as my_file:
        json.dump(node_id_pos_dict, my_file)

    # writing features of the edge graph
    with open("{}features.json".format(output_dir), "w") as my_file:
        json.dump(edge_graph_features, my_file)

    #storing graph images for better understanding
    #edge_graph_obj = nx.Graph()

    nx.draw(edge_graph_obj, node_color='lightblue', with_labels=True, node_size=500)
    plt.savefig("{}edge_graph.png".format(output_dir))
    plt.clf()
    nx.draw(g_object, node_color='lightblue', with_labels=True, node_size=500)
    plt.savefig("{}node_graph.png".format(output_dir))
    plt.clf()
    print("written to {}".format(output_dir))

def create_edge_matrix_from_weighted_node_matrix(G, g_object, flow_matrix, coutner):
    total_number_of_nodes = 0
    new_node_positions = []
    weight_for_nodes = []
    #node_weights = []

    #coverting node graph to edge graph
    for source in range(len(G)):
        for destination in range(len(G)):
            # we will have a node for every unique edge
            if G[source, destination] == 1:
                '''
                in case of undirected graph, 'A' will have both [source,destination] and [destination, source] as 1
                but we have to take unique edges, so we will take ony one of them
                '''
                if [source, destination] not in new_node_positions and [destination, source] not in new_node_positions:
                    new_node_positions.append([source, destination])
                    total_number_of_nodes += 1
                    # the edge weight of node graph will become the node weight of the edge graph
                    weight_for_nodes.append(flow_matrix[source, destination])
    #assigning node id for edge graph
    new_node_ids = [i for i in range(len(new_node_positions))]
    adj, edges = generate_adj_matrix_for_edge_graph(new_node_positions, new_node_ids)
    node_id_pos_dict = {new_node_ids[node_id]: new_node_positions[node_id] for node_id in new_node_ids}
    edge_graph_node_features = {str(node_id):weight_for_nodes[node_id] for node_id in new_node_ids }
    edge_graph_features = {"edges": edges, "features":edge_graph_node_features}

    write_data(coutner, g_object, adj, flow_matrix, edge_graph_features, node_id_pos_dict)

    # import IPython
    # IPython.embed()
    # assert False




def main():
    #creating a bidirected graph with 50 nodes
    n = 10
    # graph_obj = nxrg.barabasi_albert_graph(n, 2)
    # G = nx.adjacency_matrix(graph_obj).toarray()
    g_object = gnp_random_connected_graph(n, 0.1)
    # edge_list = [[edge[0], edge[1]] for edge in list(G.edges)]
    G = nx.adjacency_matrix(g_object).toarray()

    source = random.randint(0, n-1)
    #generating random number excluding source
    destination = random.choice([node_id for node_id in range(n) if node_id not in [source] ])
    print("Source: {}\nDestination: {}".format(source, destination))

    shortest_path, shortest_distance = fetch_shortest_path(G, source, destination, len(G))
    if shortest_path:
        print("Shortest Path: {}\nDistance: {}".format(shortest_path, shortest_distance))
        '''
            Currently we assume that there will only be 1 dominant flow
            The assign weights function always assigns random weights to each edge, while making sure that all the edges
            in the shortest path recieve maximum weight.
        '''
        flow_matrices = []
        index = 0
        while index < 50:
            # print(index)
            flow_matrix = assign_weights(G, n, shortest_path)
            flow_matrices.append(flow_matrix)
            create_edge_matrix_from_weighted_node_matrix(G, g_object, flow_matrix, index)
            index += 1
    else:
        print("path does not exist")

    # import IPython
    # IPython.embed()
    # assert False


if __name__ == "__main__":

    main()