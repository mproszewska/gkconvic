import torch
import networkx as nx
import numpy as np

from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Dataset


def to_torch_sparse_tensor(x, size=None):
    return to_dense_adj(x, max_num_nodes=size)[0].to_sparse()


class SubgraphsDataset(Dataset):
    def __init__(self, pt_file, x_dim=None, remove_features=False):
        super(SubgraphsDataset, self).__init__()
        self.data = torch.load(pt_file)
        self.node_features = False if x_dim is not None else True
        self.x_dim = self.data[0]["x"].shape[-1] if self.node_features else x_dim
        self.remove_features = remove_features

        self.subgraph_size = 1 + max(
            max(
                S["edge_index"].amax(dim=(0, 1)).item() if S["edge_index"].shape[-1] else 0
                for _, S in G["subgraphs"].items()
            )
            for _, G in self.data.items()
        )
        ys = [G["y"].item() for _, G in self.data.items()]
        self._num_classes = max(ys) + 1
        assert len(np.unique(ys)) == self._num_classes
        xs, adjs, nidxs, graph_indicators, ys = list(), list(), list(), list(), list()
        for i in range(len(self.data)):
            x, adj, nidx, y = self.preprocess(i)
            xs += [x]
            adjs += [adj.to_dense()]
            nidxs += [nidx]
            ys += [y]
            graph_indicators += [i * torch.ones(len(x))]

        self.xs, self.adjs, self.nidxs, self.ys = (
            torch.cat(xs),
            torch.cat(adjs),
            torch.cat(nidxs),
            torch.cat(ys),
        )
        if remove_features:
            self.xs = torch.ones_like(self.xs)
        self.graph_indicators = torch.cat(graph_indicators)

    @property
    def num_classes(self):
        return self._num_classes

    def len(self):
        return len(self.data)

    def get(self, idx):
        mask = self.graph_indicators == idx
        if torch.sum(mask) == 0:
            assert False, idx
        return self.xs[mask], self.adjs[mask], self.nidxs[mask], self.ys[idx : idx + 1]

    def preprocess(self, idx):
        y = self.data[idx]["y"]
        subgraphs = self.data[idx]["subgraphs"]

        if not self.node_features:
            x = torch.ones(len(subgraphs), self.x_dim)
        else:
            x = self.data[idx]["x"].to_dense()
        additional_x = torch.zeros(1, x.shape[-1])
        additional_adj = to_torch_sparse_tensor(torch.zeros(2, 0, dtype=torch.long), size=self.subgraph_size)
        additional_nidx = len(subgraphs) * torch.ones(self.subgraph_size)
        x = torch.cat((x, additional_x))
        adj = torch.stack(
            tuple(
                to_torch_sparse_tensor(subgraphs[i]["edge_index"], size=self.subgraph_size)
                for i in range(len(subgraphs))
            )
            + (additional_adj,)
        )
        key = "idx" if "idx" in subgraphs[0] else "nidx"
        nidx = torch.stack(
            tuple(subgraphs[i][key].clone().detach() for i in range(len(subgraphs))) + (additional_nidx,)
        ).long()
        return x, adj, nidx, y


def collate_fn(batch):
    n = len(batch[0])
    stacked = list()

    graph_indicator = torch.cat(
        tuple(i * torch.ones(len(b[1]), dtype=torch.long) for i, b in enumerate(batch))
    )
    for i in range(n):
        if i == n - 1:
            stacked += [graph_indicator]
        stacked += [torch.cat(tuple(b[i] for b in batch))]

    _, counts = torch.unique(graph_indicator, return_counts=True)
    x, adj, nidx, graph_indicator, y = tuple(stacked)
    counts_sum = torch.cumsum(counts, dim=0)[:-1].long()
    nidx_diff = torch.zeros_like(graph_indicator)
    nidx_diff[counts_sum.long()] = counts[:-1]
    nidx_diff = torch.cumsum(nidx_diff, dim=0)
    nidx = nidx + nidx_diff.unsqueeze(-1)
    return x, adj, nidx, graph_indicator, y


class Metric:
    def __init__(self):
        self.values = {"train": [], "val": [], "test": []}
        self.best = {"train": 0, "val": 0, "test": 0}

    def add(self, value, split):
        self.values[split] += [value.item()]

    def get(self, split):
        if len(self.values[split]):
            return np.mean(self.values[split])
        else:
            return 0

    def save_higher(self):
        updated = False
        if self.best["val"] < self.get("val"):
            self.best = {
                "train": self.get("train"),
                "val": self.get("val"),
                "test": self.get("test"),
            }
            updated = True
        self.values = {"train": [], "val": [], "test": []}
        return updated

    def restart(self):
        self.values = {"train": [], "val": [], "test": []}
        self.best = {"train": 0, "val": 0, "test": 0}

    def get_best(self, split):
        return self.best[split]


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def accuracy(pred, y):
    return (pred.argmax(dim=-1) == y).float().mean()


def save_model(path, state_dict, model_args):
    torch.save({"state_dict": state_dict, "args": model_args}, path)
