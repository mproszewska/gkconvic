import argparse
import copy
import json
import os
import logging
import time
from tqdm import tqdm

from models import GKNetwork, max_comp
from utils import *
from models import max_comp

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import DataLoader

from grakel.kernels import (
    WeisfeilerLehman,
    VertexHistogram,
    WeisfeilerLehmanOptimalAssignment,
    Propagation,
    GraphletSampling,
    RandomWalk,
    RandomWalkLabeled,
    PyramidMatch,
)

SAVE_DIR = "datasets/saved"
SPLITS_DIR = "datasets/splits"
LOG_DIR = "logs"


def args_parser():
    parser = argparse.ArgumentParser(description="graph-kernel-networks")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--save_path", type=str, default=None, help="model save path")
    parser.add_argument("--load_path", type=str, default=None, help="model load path")
    parser.add_argument("--just_eval", action="store_true", default=False, help="Just run evaluation")

    parser.add_argument("--split", type=int, default=-1, help="Split index")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument(
        "--early_stop", type=int, default=40, help="number of evaluations before early stopping",
    )
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32], help="hidden layers sizes")
    parser.add_argument(
        "--mlp_hidden_dims", type=int, nargs="+", default=[], help="hidden MLP layers sizes",
    )

    parser.add_argument(
        "--filters_sizes", nargs="+", type=int, default=[6, 6], help="number of nodes in filters",
    )
    parser.add_argument("--x_dim", type=int, default=None, help="number of features")
    parser.add_argument("--max_step", type=int, default=1, help="max length of random walks")
    parser.add_argument("--encoder_dim", type=int, default=16, help="dim of features encoder")
    parser.add_argument("--dropout_rate", type=float, default=0.4, help="dropout rate")

    parser.add_argument("--eval_freq", type=int, default=10, help="frequency of test evaluation")
    parser.add_argument("--seed", type=int, default=0, help="seed for splitting the dataset")
    parser.add_argument("--pass_x", action="store_true", default=False, help="whether to pass x to MLP")
    parser.add_argument(
        "--remove_features", action="store_true", default=False, help="whether to remove features",
    )
    parser.add_argument(
        "--optimizer",
        default="scheduler",
        help="optimizer version",
        choices=["scheduler", "two_lrs", "finetune_mlp"],
    )
    parser.add_argument(
        "--kernel",
        default="rw",
        help="Kernel name",
        choices=["rw", "dwr", "wl", "rwl", "wloa", "prop", "gl", "py"],
    )
    parser.add_argument(
        "--gk_layer_type", default="diff", help="Type of GK layer", choices=["non_diff", "diff"],
    )
    parser.add_argument(
        "--pool_fn", nargs="+", default=["add"], help="Pooling function", choices=["add", "max", "mean"],
    )
    parser.add_argument(
        "--ker_activation",
        nargs="+",
        default=[],  # "batch_norm", "relu"],
        help="Function after Kernel block",
        choices=["mask", "batch_norm", "relu", "scale"],
    )
    parser.add_argument("--jsd_weight", type=float, default=0, help="JDS regularization weight")
    parser.add_argument("--mlp_weight", type=float, default=0, help="MLP regularization weight")
    parser.add_argument("--contr_weight", type=float, default=0, help="Contrastive loss weight")
    parser.add_argument("--fix_mlp", action="store_true", default=False, help="whether to fix MLP")
    parser.add_argument(
        "--pos_connection", type=float, default=-1.0, help="Positive connection for fixed MLP",
    )
    parser.add_argument(
        "--neg_connection", type=float, default=0.5, help="Negative connection for fixed MLP",
    )
    parser.add_argument(
        "--normalize_kernel",
        action="store_true",
        default=False,
        help="whether to normalize non diff kernel outputs",
    )
    parser.add_argument(
        "--activation", default="relu", help="Activation function for MLP", choices=["relu", "sigmoid"],
    )
    args = parser.parse_args()
    return args


def get_logger(log_filename):
    log_filename = os.path.join(LOG_DIR, log_filename)
    print(log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_optimizer(model, args):
    if args.fix_mlp:
        model.mlp.init_weights(
            len(args.pool_fn), pos_connection=args.pos_connection, neg_connection=args.neg_connection,
        )
    if args.optimizer == "scheduler":
        optimizer = torch.optim.Adam(list(set(model.parameters())), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.optimizer == "two_lrs":
        graph_params = set(model.ker_layers.parameters())
        cla_params = set(model.parameters()) - graph_params
        optimizer = torch.optim.Adam(
            [
                {"params": list(graph_params), "lr": args.lr},
                {"params": list(cla_params), "lr": args.lr * 0.1},
            ]
        )
        scheduler = None
    elif args.optimizer == "finetune_mlp":
        mlp_params = set(model.mlp.parameters())
        optimizer = torch.optim.Adam([{"params": list(mlp_params), "lr": args.lr},])
        scheduler = None
    else:
        raise NotImplementedError(f"Unsupported optimizer {args.optimizer}")
    return optimizer, scheduler


def main():
    args = args_parser()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = SubgraphsDataset(args.dataset_path, x_dim=args.x_dim, remove_features=args.remove_features)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    np.random.seed(args.seed)
    perm = torch.randperm(len(dataset), generator=generator)

    if args.split < 0:
        val_idx, test_idx = int(len(dataset) * 0.8), int(len(dataset) * 0.9)
        train_idxs, val_idxs, test_idxs = (
            perm[:val_idx],
            perm[val_idx:test_idx],
            perm[test_idx:],
        )
    else:
        dataset_name = args.dataset_path.split("/")[2]
        with open(os.path.join(SPLITS_DIR, f"{dataset_name}.json")) as f:
            splits = json.load(f)
        split = splits[args.split]
        test_idxs = split["test"]
        train_idxs, val_idxs = (
            split["model_selection"][0]["train"],
            split["model_selection"][0]["validation"],
        )

    dataset_train = torch.utils.data.Subset(dataset, train_idxs)
    dataset_val = torch.utils.data.Subset(dataset, val_idxs)
    dataset_test = torch.utils.data.Subset(dataset, test_idxs)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    kernel_kwargs = {"kernel": args.kernel}
    if args.gk_layer_type == "diff":
        kernel_kwargs = dict(
            **kernel_kwargs,
            **{
                "max_step": args.max_step,
                "encoder_dim": args.encoder_dim,
                "dropout_rate": args.dropout_rate,
                "num_labels": dataset.x_dim,
            },
        )
    elif args.gk_layer_type == "non_diff":
        kernel_kwargs["normalize"] = args.normalize_kernel
    else:
        raise ValueError(f"Invalid GK layer type {args.gk_layer_type}")
    model_args = {
        "gk_layer_type": args.gk_layer_type,
        "in_features": dataset.x_dim,
        "out_features": dataset.num_classes,  # if dataset.num_classes > 2 else 1,
        "hidden_dims": args.hidden_dims,
        "mlp_hidden_dims": args.mlp_hidden_dims,
        "kernel_kwargs": kernel_kwargs,
        "filters_sizes": args.filters_sizes,
        "dropout_rate": args.dropout_rate,
        "pass_x": args.pass_x,
        "pool_fn": args.pool_fn,
        "jsd_weight": args.jsd_weight,
        "ker_activation": args.ker_activation,
        "mlp_weight": args.mlp_weight,
        "contr_weight": args.contr_weight,
        "activation": args.activation,
    }
    model = GKNetwork(**model_args).to(device)

    if args.load_path is not None:
        params = torch.load(args.load_path, map_location=device)
        model = GKNetwork(**params["args"])
        model.load_state_dict(params["state_dict"])
        model.eval()
        model = model.to(device)

    optimizer, scheduler = get_optimizer(model, args)
    early_stopper = EarlyStopper(patience=args.early_stop, min_delta=0.00)
    bce = torch.nn.BCEWithLogitsLoss()

    def train(model, optimizer, x, adj, nidx, graph_indicator, y):
        x, adj, nidx, graph_indicator = (
            x.to(device),
            adj.to(device),
            nidx.to(device),
            graph_indicator.to(device),
        )
        y = y.to(device)
        optimizer.zero_grad()
        output, _, loss = model(x, adj, nidx, graph_indicator, y)

        if output.shape[-1] == 1:
            loss = sum(loss) + bce(output[:, 0], y.float())
        else:
            loss = sum(loss) + F.cross_entropy(output, y)
        acc = accuracy(output, y)
        loss.backward()
        optimizer.step()
        return loss, acc

    def test(model, x, adj, nidx, graph_indicator, y):
        x, adj, nidx, graph_indicator = (
            x.to(device),
            adj.to(device),
            nidx.to(device),
            graph_indicator.to(device),
        )
        y = y.to(device)
        output, _, loss = model(x, adj, nidx, graph_indicator, y)
        if output.shape[-1] == 1:
            loss = sum(loss) + bce(output[:, 0], y.float())
        else:
            loss = sum(loss) + F.cross_entropy(output, y)
        acc = accuracy(output, y)
        return loss, acc

    time_str = time.strftime("%Y%m%d_%H%M%S")
    logger = get_logger(f"{time_str}.log")
    logger.info(args)
    logger.info(model)

    pbar = tqdm(range(1, args.epochs + 1), total=args.epochs, bar_format="{l_bar}{bar:500}{r_bar}{bar:-10b}",)
    loss, acc = Metric(), Metric()
    bast_state_dict = copy.deepcopy(model.state_dict())
    for epoch in pbar:
        description_str = f"epoch={epoch},"

        if not args.just_eval:
            model.train()

            for x, adj, nidx, graph_indicator, y in dataloader_train:
                batch_loss, batch_acc = train(model, optimizer, x, adj, nidx, graph_indicator, y)
                loss.add(batch_loss, "train"), acc.add(batch_acc, "train")
            description_str += f"train_loss={loss.get('train'):.4f},train_acc={acc.get('train'):.4f},"

        if epoch % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                for x, adj, nidx, graph_indicator, y in dataloader_val:
                    batch_loss, batch_acc = test(model, x, adj, nidx, graph_indicator, y)
                    loss.add(batch_loss, "val"), acc.add(batch_acc, "val")
                description_str += f"val_loss={loss.get('val'):.4f},val_acc={acc.get('val'):.4f},"

                for x, adj, nidx, graph_indicator, y in dataloader_test:
                    batch_loss, batch_acc = test(model, x, adj, nidx, graph_indicator, y)
                    loss.add(batch_loss, "test"), acc.add(batch_acc, "test")
                description_str += f"test_loss={loss.get('test'):.4f},test_acc={acc.get('test'):.4f}"
                pbar.set_description_str(description_str, refresh=True)
                if early_stopper.early_stop(loss.get("val")):
                    break

        loss.restart()
        if acc.save_higher():
            bast_state_dict = copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()
        logger.info(description_str)
    logger.info(
        f"Final:train_acc={acc.get_best('train'):.4f},val_acc={acc.get_best('val'):.4f},test_acc={acc.get_best('test'):.4f}"
    )
    if args.save_path is not None:
        save_model(args.save_path, bast_state_dict, model_args)


if __name__ == "__main__":
    main()
