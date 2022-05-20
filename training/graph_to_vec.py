"""Graph2Vec module."""

import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def path2name(path):
    base = os.path.basename(path)
    name = os.path.basename((path.split(base)[0])[0:-1])
    return name
    #below was the orignal code
    #base = os.path.basename(path)
    #return os.path.splitext(base)[0]


def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)

    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])
    if "features" in data.keys():
        features = data["features"]
        features = {int(k): v for k, v in features.items()}
    else:
        features = nx.degree(graph)
        features = {int(k): v for k, v in features}

    return graph, features, name


def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    # import IPython
    # IPython.embed()
    # assert False
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc


def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = path2name(f)
        out.append([identifier] + list(model.dv["g_" + identifier]))
        #out.append([identifier] + list(model.docvecs["g_" + identifier]))
    column_names = ["type"] + ["x_" + str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)


def fetch_feature_data(inp_path):
    files = ["{}{}/features.json".format(inp_path, dir) for dir in os.listdir(inp_path) if \
             os.path.exists("{}{}/features.json".format(inp_path, dir))]
    print("total files: {}".format(len(files)))
    return files


def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """

    graphs = fetch_feature_data(args.input_path)
    #graphs = glob.glob(os.path.join(args.input_path, "*.json"))
    print("\nFeature extraction started.\n")
    #document = feature_extractor(graphs[0], args.wl_iterations)
    document_collections = Parallel(n_jobs=args.workers)(\
        delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")
    # import IPython
    # IPython.embed()
    # assert False
    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)

    save_embedding(args.output_path, model, graphs, args.dimensions)


if __name__ == "__main__":
    args = parameter_parser()
    main(args)