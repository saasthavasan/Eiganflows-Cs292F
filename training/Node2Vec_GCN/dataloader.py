import torch

class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, inputs, labels):
        """Initialization"""
        self.labels = labels
        self.inputs = inputs
        self.edges = list(self.labels.keys())

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs)

    def __getitem__(self, index):
        """Generates one sample of data"""

        return torch.tensor(self.inputs[index], dtype=torch.long), \
            [torch.tensor(self.labels[edges][index]) for edges in self.edges]
