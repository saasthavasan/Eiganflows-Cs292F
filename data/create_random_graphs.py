import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx.generators.random_graphs as nxrg

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
        print("{},{}, {}".format(shortest_path[i], shortest_path[i + 1], wt))

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


def main():
    #creating a bidirected graph with 50 nodes
    n = 100
    graph_obj = nxrg.barabasi_albert_graph(n, 2)
    G = nx.adjacency_matrix(graph_obj).toarray()

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
            flow_matrix = assign_weights(G, n, shortest_path)
            flow_matrices.append(flow_matrix)
    else:
        print("path does not exist")

    import IPython
    IPython.embed()
    assert False


if __name__ == "__main__":

    main()