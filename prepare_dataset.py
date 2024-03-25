import argparse
import json
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric import utils
from torch_geometric.datasets import BA2MotifDataset, TUDataset
from torch_geometric.utils import to_torch_sparse_tensor, to_dense_adj

TU_DATASETS = [
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "COLLAB",
    "PROTEINS",
    "MUTAG",
    "ENZYMES",
    "NCI1",
    "DD",
    "PTC_MR",
]
SYNTHETIC_DATASETS = ["BA2Motif"]
SAVE_DIR = "datasets/saved"


def args_parser():
    parser = argparse.ArgumentParser(description="graph-kernel-networks")
    parser.add_argument(
        "--dataset", default="IMDB-BINARY", help="dataset name", choices=TU_DATASETS + SYNTHETIC_DATASETS,
    )
    parser.add_argument("--k", type=int, default=2, help="k-hop neighborhood to construct the subgraph ")
    parser.add_argument("--subgraph_size", type=int, default=10, help="max size of the subgraph")
    parser.add_argument(
        "--one_in_feature", action="store_true", default=False, help="Number of neighbours as feature",
    )
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    if args.dataset in TU_DATASETS:
        dataset = TUDataset(SAVE_DIR, args.dataset, use_node_attr=True)
    elif args.dataset in SYNTHETIC_DATASETS:
        dataset = BA2MotifDataset(os.path.join(SAVE_DIR, args.dataset))
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    def extract_subgraph(G, node):
        subset = torch.tensor([node], dtype=torch.long)
        if node not in G.edge_index:
            subset = torch.tensor([node])
            edge_index, edge_attr = torch.zeros([2, 0], dtype=torch.long), None
        else:
            for k in range(1, args.k + 1):
                s = utils.k_hop_subgraph(node, k, edge_index=G.edge_index, relabel_nodes=True)[0]
                s = torch.cat((subset, s))
                _, inverse = np.unique(s, return_index=True)
                subset = torch.tensor([s[index.item()] for index in sorted(inverse)], dtype=torch.long)
                if len(subset) > args.subgraph_size:
                    break
                assert subset[0] == node, (subset, node)
            subset = subset[: args.subgraph_size]
            edge_index, edge_attr = utils.subgraph(subset, edge_index=G.edge_index)
        if len(subset) < args.subgraph_size:
            subset = torch.cat(
                (subset, G.num_nodes * torch.ones(args.subgraph_size - len(subset), dtype=torch.long),)
            )
        map_node = {node.item(): nidx for nidx, node in enumerate(subset)}
        edge_index = edge_index.apply_(lambda node: map_node[node])
        return {"edge_index": edge_index, "nidx": subset}

    graphs = dict()
    for nidx, G in tqdm(enumerate(dataset), total=len(dataset)):
        subgraphs = {node: extract_subgraph(G, node) for node in range(G.num_nodes)}
        x = None if G.x is None else torch.cat((G.x, torch.zeros(1, G.x.shape[-1])))
        if (x is None) or (args.dataset in SYNTHETIC_DATASETS):
            x = to_dense_adj(G.edge_index)[0].sum(dim=-1)
            if not args.one_in_feature:
                x = torch.where(x > args.subgraph_size, args.subgraph_size, x)
                x = F.one_hot(x.long(), num_classes=args.subgraph_size + 1).float()
        if len(x) != G.num_nodes:
            x = x[:-1]
        graphs[nidx] = {"x": x, "subgraphs": subgraphs, "y": G.y}

    torch.save(
        graphs, os.path.join(SAVE_DIR, args.dataset, f"ss{args.subgraph_size}_k{args.k}.pt"),
    )


if __name__ == "__main__":
    main()
