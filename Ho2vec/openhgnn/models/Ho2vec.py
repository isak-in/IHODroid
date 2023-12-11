import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from dgl.nn.pytorch import GraphConv, GCN2Conv
from . import BaseModel, register_model
from ..layers.macro_layer.SemanticConv import SemanticAttention
from ..layers.MetapathConv_v4 import MetapathConv
from ..utils.utils import extract_metapaths


@register_model('Ho2vec')
class Ho2vec(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.meta_paths_dict is None:
            meta_paths = extract_metapaths(args.category, hg.canonical_etypes)
        else:
            meta_paths = args.meta_paths_dict
            
        return cls(meta_paths=meta_paths, category=args.out_node_type,
                   in_size=args.in_dim, hidden_size=args.hidden_dim,
                   out_size=args.out_dim,
                   n_layers=args.n_layers,
                   dropout=args.dropout)

    def __init__(self, meta_paths, category, in_size, hidden_size, out_size, n_layers, dropout):
        super(Ho2vec, self).__init__()
        self.category = category
        self.layers = nn.ModuleList()
        self.layers.append(HGCNLayer(meta_paths, in_size, hidden_size, n_layers, dropout))

        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, g, h_dict):

        for gnn in self.layers:
            h_dict = gnn(g, h_dict)

        out_dict = {ntype: self.linear(h_dict[ntype]) for ntype in self.category}

        return out_dict

    def get_emb(self, g, h_dict):
        h = h_dict[self.category]
        for gnn in self.layers:
            h = gnn(g, h)

        return {self.category: h.detach().cpu().numpy()}


class HGCNLayer(nn.Module):

    def __init__(self, meta_paths_dict, in_size, hidden_size, n_layers, dropout):
        super(HGCNLayer, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        self.gcn_layers = nn.ModuleList()
        semantic_attention = SemanticAttention(in_size=hidden_size)
        mods = nn.ModuleDict({mp: GCNIINet(in_size, hidden_size, hidden_size, n_layers) for mp in meta_paths_dict})

        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp, mp_value in self.meta_paths_dict.items():
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, mp_value)

        h = self.model(self._cached_coalesced_graph, h)

        return h


class GCNIINet(nn.Module):
    def __init__(self, in_size, hidden=64, out_size=64, num_layers=3):

        super(GCNIINet, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_size, hidden))

        for i in range(self.num_layers):
            self.convs.append(GCN2Conv(hidden, i+1, alpha=0.2, project_initial_features=True, allow_zero_in_degree=True))
            self.bns.append(nn.BatchNorm1d(hidden))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, graph, h):
        x, adj_t = h, graph


        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=0.1, training=self.training)
            x = self.convs[i+1](adj_t, x, h)
            x = self.bns[i](x)
            x = F.relu(x)


        x = F.dropout(x, p=0.1, training=self.training)
        x = self.convs[-1](adj_t, x, h)

        return x