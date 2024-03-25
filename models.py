import numpy as np

import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import dense_to_sparse, to_dense_adj

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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from scipy.cluster.vq import kmeans2


class MLP(nn.Module):
    def __init__(
        self, in_features, out_features, hidden_dims, dropout_rate=0.5, activation="relu",
    ):
        super(MLP, self).__init__()

        self.dropout_rate = dropout_rate
        dims = [in_features] + hidden_dims
        self.layers = list()
        for i in range(len(dims) - 1):
            self.layers += [
                nn.Linear(dims[i], dims[i + 1], bias=False),
                {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}[activation],
                nn.Dropout(p=dropout_rate),
            ]
        self.layers += [nn.Linear(dims[-1], out_features, bias=False)]
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

    def get_init_weights(self, pool_fn_num, pos_connection, neg_connection):
        assert len(self.layers) == 1
        layer = self.layers[0]
        layer.weight = nn.Parameter(torch.ones_like(layer.weight))
        weight = torch.zeros(pool_fn_num, layer.in_features // pool_fn_num, layer.out_features)
        n_weights_per_class = (layer.in_features // pool_fn_num) // layer.out_features
        if (layer.in_features // pool_fn_num) != n_weights_per_class * layer.out_features:
            n_weights_per_class += 1
        for j in range(layer.in_features // pool_fn_num):
            weight[:, j, j // n_weights_per_class] = 1.0
        weight = weight.reshape(-1, layer.out_features)
        pos_weight = torch.t(weight)
        neg_weight = 1 - pos_weight
        weight = pos_connection * pos_weight + neg_connection * neg_weight
        return weight

    def init_weights(self, pool_fn_num, pos_connection, neg_connection):
        assert len(self.layers) == 1
        weight = self.get_init_weights(pool_fn_num, pos_connection, neg_connection)
        self.layers[0].weight.data.copy_(weight)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self.reset = False

        self.codebook_init = False

    def reset_codebook(self, x):
        if self.codebook_init:
            centroid, label = kmeans2(
                x.detach().cpu().numpy(), self._embedding.weight.detach().cpu().numpy(), minit="matrix",
            )
        else:
            centroid, label = kmeans2(
                (x + torch.randn_like(x) * 1e-4).detach().cpu().numpy(), self._num_embeddings,
            )
        self._embedding.weight.data = torch.from_numpy(centroid).float().to(x.device)
        self.codebook_init = True

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        if self.training:
            self.reset_codebook(flat_input)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized_ind = encoding_indices.view(input_shape[:-1])
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(
                encodings, 0
            )

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), quantized_ind, perplexity, encodings


def max_comp(E, d):
    E = list(E)

    if len(E) == 0:
        return E, d
    return E, d

    graph = csr_matrix((np.ones(len(E)), zip(*E)), [np.max(E) + 1] * 2)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    (unique, counts) = np.unique(labels, return_counts=True)
    max_elms = np.argwhere(labels == unique[np.argmax(counts)])

    max_ed_list = [e for e in E if (e[0] in max_elms) and (e[1] in max_elms)]

    dnew = dict([((int(k), d[k])) for k in max_elms.flatten()])
    return max_ed_list, dnew


class NonDiffGKernel(nn.Module):
    def __init__(self, nodes, labels, filters=8, kernel="wl", normalize=True, store_fit=False):
        super(NonDiffGKernel, self).__init__()

        A = torch.from_numpy(np.random.rand(filters, nodes, nodes)).float()
        A = ((A + A.transpose(-2, -1)) > 1).float()
        A = torch.stack([a - torch.diag(torch.diag(a)) for a in A], 0)
        self.P = nn.Parameter(A, requires_grad=False)

        self.X = nn.Parameter(
            torch.stack([F.one_hot(torch.randint(labels, (nodes,)), labels) for fi in range(filters)], 0,),
            requires_grad=False,
        )

        self.Xp = nn.Parameter(torch.zeros((filters, nodes, labels)).float(), requires_grad=True)

        self.Padd = nn.Parameter(torch.randn(filters, nodes, nodes) * 0)
        self.Prem = nn.Parameter(torch.randn(filters, nodes, nodes) * 0)
        self.Padd.data = self.Padd.data + self.Padd.data.transpose(-2, -1)
        self.Prem.data = self.Prem.data + self.Prem.data.transpose(-2, -1)

        self.filters = filters
        self.store = [None] * filters

        self.gks = []
        for k in kernel.split("+"):
            self.gks.append(grakel_kernel(k, normalize))

        self.store_fit = store_fit
        self.stored = False

    def forward(self, x, adj, nidx, not_used=None, fixedges=None, node_indexes=[]):
        convs = []
        for gk in self.gks:
            convs.append(
                NonDiffGKConv.apply(
                    x,
                    adj,
                    nidx,
                    self.P,
                    self.Padd,
                    self.Prem,
                    self.X,
                    self.Xp,
                    self.training,
                    gk(None),
                    self.stored,
                    node_indexes,
                )
            )
        conv = torch.cat(convs, -1)
        return conv


class NonDiffGKConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, adj, nidx, P, Padd, Prem, X, Xp, training, gk, stored, node_indexes):
        # graph similarity here
        if not stored:
            egonets = list()
            for i in range(adj.shape[0]):
                x_i = x[nidx[i]]
                adj_i = adj[i]
                edge_index_i = dense_to_sparse(adj_i)[0].long()
                egonets += [(x_i, edge_index_i)]
            G1 = lambda i: [
                set([(e[0], e[1]) for e in egonets[i][1].t().cpu().numpy()]),
                dict(zip(range(egonets[i][0].shape[0]), egonets[i][0].argmax(-1).cpu().numpy(),)),
            ]
            Gs1 = [G1(i) for i in range(adj.shape[0])]
            for g in Gs1:
                if len(g[0]) == 0:
                    g[0] = {(len(g[1]) - 1, len(g[1]) - 1)}
            conv = NonDiffGKConv.eval_kernel(x, Gs1, P, X, gk, False)
        else:
            conv = NonDiffGKConv.eval_kernel(None, None, P, X, gk, True)[nidx, :]
            Gs1 = None
        conv = conv.to(x.device)
        ctx.save_for_backward(x, adj, P, Padd, Prem, X, Xp, conv)
        ctx.stored = stored
        ctx.node_indexes = nidx
        ctx.Gs1 = Gs1
        ctx.P = P
        ctx.X = X
        ctx.gk = gk

        return conv.float()

    @staticmethod
    def backward(ctx, grad_output):
        x, edge_index, P, Padd, Prem, X, Xp, conv = ctx.saved_tensors
        P = ctx.P

        # grad_input -> kernel response gradient size: filters x nodes
        # todo: estimate gradient and keep the one maximizing dot product

        # perform random edit for each non zero filter gradient:
        grad_padd = 0
        grad_prem = 0
        grad_xp = 0

        kindexes = torch.nonzero(torch.norm(grad_output, dim=0))[:, 0]
        Pnew = P.clone()
        Xnew = X.clone()

        for i in range(3):  # if the gradient of the edit w.r.t. the loss is 0 we try another edit operation
            for fi in kindexes:
                edit_graph = torch.rand((1,)).item() < 0.5 or X.shape[-1] == 1
                Pnew, Xnew = NonDiffGKConv.random_edit(fi, Pnew, Padd, Prem, Xnew, Xp, edit_graph)
            if not ctx.stored:
                convnew = NonDiffGKConv.eval_kernel(x, ctx.Gs1, Pnew, Xnew, ctx.gk, True)
            else:
                convnew = NonDiffGKConv.eval_kernel(None, None, Pnew, Xnew, ctx.gk, True)[ctx.node_indexes, :]
            convnew = convnew.to(conv.device)
            grad_fi = conv - convnew

            proj = (grad_fi * grad_output).sum(0)[:, None, None]
            kindexes = kindexes[proj[kindexes, 0, 0] == 0]
            if len(kindexes) == 0:
                break

        grad_padd += proj * (P - Pnew)
        grad_prem += proj * (Pnew - P)
        grad_xp += proj * (X - Xnew)

        th = 0
        ctx.P.data = (proj >= th) * Pnew + (proj < th) * P
        ctx.X.data = (proj >= th) * Xnew + (proj < th) * X

        return (
            None,
            None,
            None,
            None,
            grad_padd * ((Padd).sigmoid() * (1 - (Padd).sigmoid())),
            grad_prem * ((Prem).sigmoid() * (1 - (Prem).sigmoid())),
            None,
            grad_xp * (Xp.sigmoid() * (1 - Xp.sigmoid())),
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def eval_kernel(x, Gs1, P, X, gk, stored=False):
        filters = P.shape[0]
        nodes = P.shape[1]
        Gs2 = [
            max_comp(
                set(
                    [(e[0], e[1]) for e in torch_geometric.utils.dense_to_sparse(P[fi])[0].t().cpu().numpy()]
                ),
                dict(zip(range(nodes), X[fi].argmax(-1).flatten().detach().cpu().numpy())),
            )
            for fi in range(filters)
        ]
        idx = np.array([len(a) > 0 for a, b in Gs2])

        if not stored:
            gk.fit(Gs1)
            sim = gk.transform([g for g in Gs2 if len(g[0]) > 0])
            sim = np.nan_to_num(sim)
            sim_zeros = np.zeros((len(Gs2), len(Gs1)))
            sim_zeros[idx] = sim
            sim = sim_zeros
        else:
            sim = gk.transform([g for g in Gs2 if len(g[0]) > 0])
            sim = np.nan_to_num(sim)
            sim_zeros = np.zeros((len(Gs2), sim.shape[1]))
            sim_zeros[idx] = sim
            sim = sim_zeros

        return torch.from_numpy(sim.T)

    @staticmethod
    def random_edit(i, Pin, Padd, Prem, X, Xp, edit_graph, n_edits=1, temp=0.1):
        P = Pin.clone()
        X = X.clone()
        if edit_graph:  # edit graph
            Pmat = (
                P[i] * (Prem[i] * temp).sigmoid().data + (1 - P[i]) * (Padd[i] * temp).sigmoid().data + 1e-8
            )  # sample edits
            Pmat = Pmat * (1 - torch.eye(Pmat.shape[-1], device=Pmat.device))
            Pmat = torch.nan_to_num(Pmat)
            Pmat = Pmat / Pmat.sum()
            inds = np.random.choice(
                Pmat.shape[0] ** 2, size=(n_edits,), replace=False, p=Pmat.flatten().cpu().numpy(),
            )
            inds = torch.from_numpy(np.stack(np.unravel_index(inds, Pmat.shape), 0)).to(Pmat.device)

            inds = torch.cat([inds, inds[[1, 0], :]], -1)  # symmetric edit
            P[i].data[inds[0], inds[1]] = 1 - P[i].data[inds[0], inds[1]]

            if P[i].sum() == 0:  # avoid fully disconnected graphs
                P = Pin.clone()
        else:  # edit labels
            PX = (Xp[i].cpu() * temp).softmax(-1).data
            pi = 1 - PX.max(-1)[0]
            pi = pi / pi.sum(-1, keepdims=True)
            X = torch.nan_to_num(X)
            pi = torch.nan_to_num(pi)
            pi = pi / pi.sum()
            lab_ind = np.random.choice(X[i].shape[0], (n_edits,), p=pi.cpu().numpy())
            lab_val = [
                np.random.choice(PX.shape[1], size=(1,), replace=False, p=PX[j, :].numpy(),) for j in lab_ind
            ]
            lab_ind, lab_val = (
                torch.from_numpy(lab_ind),
                torch.from_numpy(np.array(lab_val)),
            )
            X[i][lab_ind, :] = 0
            X[i][lab_ind, lab_val] = 1

        return P, X


def rw_kernel(x_a, adj_a, x_b, adj_b, max_step, dropout, agg_fn=None):
    device = x_a.device
    filters_size = x_b.shape[0]
    xx = torch.einsum("mcn,abc->ambn", (x_b, x_a))  # (#G, #Nodes_filter, #Nodes_sub, D_out)
    out = []
    for i in range(max_step):
        if i == 0:
            eye = torch.eye(filters_size, device=device)
            o = torch.einsum("ab,bcd->acd", (eye, x_b))
            t = torch.einsum("mcn,abc->ambn", (o, x_a))
        else:
            x_a = torch.einsum("abc,acd->abd", (adj_a, x_a))
            x_b = torch.einsum("abd,bcd->acd", (adj_b, x_b))  # adj_hidden_norm: (Nhid,Nhid,Dout)
            t = torch.einsum("mcn,abc->ambn", (x_b, x_a))
        t = dropout(t)
        t = torch.mul(xx, t)  # (#G, #Nodes_filter, #Nodes_sub, D_out)
        t = agg_fn(t)
        out += [t]
    return sum(out) / len(out)


def drw_agg_fn(t, layer):
    t = t.permute(0, 3, 1, 2).reshape(t.shape[0], t.shape[3], -1)
    return layer(t).squeeze()


def rw_agg_fn(t):
    return torch.mean(t, dim=[1, 2])


class DiffGKLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        filters_size,
        kernel,
        dropout_rate,
        max_step=1,
        encoder_dim=16,
        subgraph_size=None,
        num_labels=1,
    ):
        super(DiffGKLayer, self).__init__()
        assert kernel in ["rw", "drw"]
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(p=dropout_rate)
        self.filters_size = filters_size
        self.max_step = max_step
        if kernel == "drw":
            self.rw_layer = torch.nn.Linear(filters_size * subgraph_size, 1)
            drw_agg_fn_with_layer = lambda x: drw_agg_fn(x, self.rw_layer)
        else:
            drw_agg_fn_with_layer = None
        self.agg_fn = {"rw": rw_agg_fn, "drw": drw_agg_fn_with_layer}[kernel]
        self.encoder = torch.nn.Linear(in_features, encoder_dim)
        self._x_hidden = Parameter(torch.FloatTensor(filters_size, encoder_dim, out_features))

        self._adj_hidden = Parameter(
            torch.FloatTensor((filters_size * (filters_size - 1)) // 2, out_features)
        )

        self.init_weights()

    def init_weights(self):
        self._adj_hidden.data.uniform_(-1, 1)
        self._x_hidden.data.uniform_(0, 1)

    def adj_hidden(self, permuted=False):
        device = self._adj_hidden.device
        adj_hidden_norm = torch.zeros(self.filters_size, self.filters_size, self.out_features).to(device)
        indices = torch.triu_indices(self.filters_size, self.filters_size, 1).to(device)
        adj_hidden_norm[indices[0], indices[1], :] = F.relu(self._adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 0, 1)
        return adj_hidden_norm if permuted else torch.permute(adj_hidden_norm, (2, 0, 1))

    def x_hidden(self, permuted=False):
        return self._x_hidden if permuted else torch.permute(self._x_hidden, (2, 0, 1))

    def forward(self, x, adj, nidx):
        device = adj.device
        x_hidden = self.x_hidden(permuted=True)
        x = F.relu(self.encoder(x.to_dense()))  # (#G, D_hid)
        adj_hidden = self.adj_hidden(permuted=True).to(device)
        adj = adj.to_dense()
        x = x[nidx]  # (#G, #Nodes_sub, D_hid)

        return rw_kernel(x, adj, x_hidden, adj_hidden, self.max_step, self.dropout, self.agg_fn,)


class NonDiffGKLayer(nn.Module):
    def __init__(
        self, in_features, out_features, filters_size, kernel, normalize=True, vq_features=None,
    ):
        super(NonDiffGKLayer, self).__init__()
        self.kernel = NonDiffGKernel(
            filters_size, in_features, out_features, kernel=kernel, normalize=normalize,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.filters_size = filters_size
        if vq_features is not None:
            self.vq_layer = VectorQuantizerEMA(vq_features, in_features, commitment_cost=0.25, decay=0.99)
        else:
            self.vq_layer = None

    def forward(self, x, adj, nidx):
        if self.vq_layer != None:
            vqloss, x, qidx, perplexity, _ = self.vq_layer(x)
            x = F.one_hot(qidx.long(), self.in_features)

        x = self.kernel(x, adj, nidx, node_indexes=nidx)
        return x

    def adj_hidden(self):
        return self.kernel.P.data

    def x_hidden(self):
        return self.kernel.X.data


class GKNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        gk_layer_type="diff",  # non_diff, diff
        hidden_dims=None,
        mlp_hidden_dims=None,
        kernel_kwargs=None,
        filters_sizes=None,
        pool_fn=None,  # 'add', 'max', 'mean'
        dropout_rate=0.4,
        pass_x=False,
        jsd_weight=0,
        ker_activation=None,
        mlp_weight=0,
        contr_weight=0,
        activation="relu",
    ):
        super(GKNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_fn = [
            {"add": global_add_pool, "max": global_max_pool, "mean": global_mean_pool}[p] for p in pool_fn
        ]
        self.pass_x = pass_x
        self.ker_activation = ker_activation

        h_fn = lambda x: -torch.sum(x * x.log(), -1)
        jsd_fn = lambda x: h_fn(x.mean(0)) - h_fn(x).mean(0)
        self.regularizer = (
            lambda x: -(jsd_weight * jsd_fn(x.softmax(-1)))
            if jsd_weight > 0
            else torch.zeros([], device=x.device)
        )

        self.ker_layers = nn.ModuleList()
        gk_layer_cls = {"non_diff": NonDiffGKLayer, "diff": DiffGKLayer}[gk_layer_type]

        self.ker_layers += [
            gk_layer_cls(
                in_features=in_features,
                out_features=hidden_dims[0],
                filters_size=filters_sizes[0],
                **kernel_kwargs,
            )
        ]

        mlp_dim = in_features if pass_x else 0
        mlp_dim += sum(hidden_dims)
        if "mask" in self.ker_activation:
            self.masks = [nn.Parameter(torch.ones(hidden_dims[0]).float(), requires_grad=False)]

        if "batch_norm" in self.ker_activation:
            self.batch_norms = torch.nn.ModuleList([nn.BatchNorm1d(hidden_dims[0])])

        dims = [in_features] + hidden_dims
        for i in range(2, len(dims)):
            additional_kernel_kwargs = {"vq_features": dims[i - 1]} if gk_layer_type == "non_diff" else {}
            self.ker_layers += [
                gk_layer_cls(
                    in_features=dims[i - 1],
                    out_features=dims[i],
                    filters_size=filters_sizes[i - 1],
                    **kernel_kwargs,
                    **additional_kernel_kwargs,
                )
            ]
            if "mask" in self.ker_activation:
                self.masks += [nn.Parameter(torch.ones(dims[i]).float(), requires_grad=False)]

            if "batch_norm" in self.ker_activation:
                self.batch_norms += [nn.BatchNorm1d(dims[i])]

        self.apply_mask = lambda x, mask: x * mask[None, :].repeat(1, x.shape[-1] // mask.shape[-1])

        self.mlp = MLP(mlp_dim * len(self.pool_fn), out_features, mlp_hidden_dims, dropout_rate, activation,)

        def arange(start, end):
            if isinstance(start, torch.Tensor) and isinstance(end, torch.Tensor):
                return torch.stack([torch.arange(s, e) for s, e in zip(start, end)])
            elif isinstance(start, torch.Tensor):
                return [torch.arange(s, end) for s in start]
            elif isinstance(end, torch.Tensor):
                return [torch.arange(start, e) for e in end]
            else:
                assert False

        self.class_filters, self.other_class_filters = dict(), dict()
        for layer in self.ker_layers:
            num_filters = layer.out_features
            num_filters_per_class = num_filters // out_features
            possible_classes = torch.arange(out_features)
            class_filters = arange(
                possible_classes * num_filters_per_class, (possible_classes + 1) * num_filters_per_class,
            )
            before_filters = arange(0, possible_classes * num_filters_per_class)
            after_filters = arange((possible_classes + 1) * num_filters_per_class, num_filters)
            other_class_filters = torch.stack(
                [torch.cat([b, a]) for b, a in zip(before_filters, after_filters)]
            )
            self.class_filters[num_filters] = class_filters
            self.other_class_filters[num_filters] = other_class_filters

        def split_responses(x, y):
            response = x.reshape(x.shape[0], -1)
            num_filters = response.shape[-1]
            self.class_filters[num_filters] = self.class_filters[num_filters].to(x.device)
            self.other_class_filters[num_filters] = self.other_class_filters[num_filters].to(x.device)
            class_filters = self.class_filters[num_filters][y]
            other_class_filters = self.other_class_filters[num_filters][y]
            response00 = torch.stack([response[j][class_filters[j]] for j in range(response.shape[0])])
            response01 = torch.stack([response[j][other_class_filters[j]] for j in range(response.shape[0])])
            return response00, response01

        if mlp_weight > 0:

            def mlp_loss(responses, y):
                assert self.pool_fn[2] == global_max_pool
                responses = responses.reshape(responses.shape[0], len(self.pool_fn), -1)[:, 2]
                response00, response01 = split_responses(responses, y)
                response00 = response00.amax(-1)
                response01 = response01.amax(-1)
                return mlp_weight * (-response00.sum() + response01.sum()) / responses.shape[0]

            self.mlp_loss = mlp_loss
        else:
            self.mlp_loss = lambda responses, y: torch.zeros([], device=self.mlp.layers[0].weight.data.device)

        if contr_weight > 0:

            def contr_loss(responses, y, graph_indicator):
                assert len(responses) == 1
                responses = responses[0]
                all_x_classes = y[graph_indicator]
                response00, response01 = split_responses(responses, all_x_classes)
                pos_dot = global_max_pool(response00, graph_indicator).amax(-1, keepdims=True)
                neg_dot = response01
                t = 1.0
                pos_logits = pos_dot / t
                neg_logits = neg_dot / t
                pos_exp, neg_exp = (
                    torch.exp(pos_logits + 1e-6),
                    torch.exp(neg_logits + 1e-6),
                )
                loss = -torch.log(
                    pos_exp / (global_add_pool(neg_exp, graph_indicator).sum(-1, keepdims=True) + pos_exp)
                )

                return contr_weight * loss.mean()

            self.contr_loss = contr_loss

        else:
            self.contr_loss = lambda responses, y, graph_indicator: torch.zeros(
                [], device=self.mlp.layers[0].weight.data.device
            )

    def forward(self, x, adj, nidx, graph_indicator, y):
        device = x.device
        unique, counts = torch.unique(graph_indicator, return_counts=True)
        n_graphs = unique.max() + 1

        responses = [x] if self.pass_x else []
        h = x

        for layer_idx in range(len(self.ker_layers)):
            h = self.ker_layers[layer_idx](h, adj, nidx)
            for ker_act in self.ker_activation:
                if ker_act == "mask":
                    h = self.apply_mask(h, self.masks[layer_idx].to(device))
                elif ker_act == "batch_norm":
                    h = self.batch_norms[layer_idx](h)
                elif ker_act == "relu":
                    h = F.relu(h)
                elif ker_act == "scale":
                    h = h / (h.sum(-1, keepdims=True) + 1e-5)
                else:
                    raise NotImplementedError(f"Invalid kernel activation {ker_act}")

            responses += [h]

        x = torch.cat(responses, -1)
        pooled = torch.zeros(n_graphs, 0, dtype=torch.float, device=device)
        for pool_fn in self.pool_fn:
            pooled = torch.cat((pooled, pool_fn(x, graph_indicator)), dim=-1)
        loss_jsd = torch.stack([self.regularizer(x) for x in responses]).mean()
        loss_jsd = torch.nan_to_num(loss_jsd)
        outputs = self.mlp(pooled)
        loss_mlp = self.mlp_loss(pooled, y)
        loss_contr = self.contr_loss(responses, y, graph_indicator)
        return outputs, responses, (loss_jsd, loss_mlp, loss_contr)


def grakel_kernel(kernel_name, normalize):
    if kernel_name == "wl":
        return lambda x: WeisfeilerLehman(n_iter=3, normalize=normalize)
    elif kernel_name == "wloa":
        return lambda x: WeisfeilerLehmanOptimalAssignment(n_iter=3, normalize=normalize)
    elif kernel_name == "prop":
        return lambda x: Propagation(normalize=normalize)
    elif kernel_name == "rwl":
        return lambda x: RandomWalkLabeled(normalize=normalize)
    elif kernel_name == "rw":
        return lambda x: RandomWalk(normalize=normalize)
    elif kernel_name == "gl":
        return lambda x: GraphletSampling(normalize=normalize)
    elif kernel_name == "py":
        return lambda x: PyramidMatch(normalize=normalize)
    else:
        return None
