import torch
from scipy.spatial import cKDTree as KDTree
import numpy as np


class GFN(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        if type(in_features) is int and type(out_features) is int:
            print(
                "Warning: no graphical data provided to GFN layer. Behaves like a standard Linear layer."
            )

        if type(in_features) is not int:
            self.in_tree = KDTree(in_features)
            self.in_graph = in_features
            in_features = in_features.shape[0]
        else:
            self.in_tree = None
            self.in_graph = None

        if type(out_features) is not int:
            self.out_tree = KDTree(out_features)
            self.out_graph = out_features
            out_features = out_features.shape[0]
        else:
            self.out_tree = None
            self.out_graph = None

        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x, in_graph=None, out_graph=None):
        if in_graph is None and out_graph is None:
            return super().forward(x)
        elif in_graph is not None and self.in_tree is None:
            raise ValueError(
                "Input graphical data provided but GFN layer was not initialized with input graphical data."
            )
        elif out_graph is not None and self.out_tree is None:
            raise ValueError(
                "Output graphical data provided but GFN layer was not initialized with output graphical data."
            )

        weight = self.weight
        bias = self.bias
        in_tree = self.in_tree
        out_tree = self.out_tree
        original_in_graph = self.in_graph
        original_out_graph = self.out_graph
        new_in_graph = in_graph
        new_out_graph = out_graph

        # -- ENCODER-style --
        if new_in_graph is not None:
            new_kd_tree = KDTree(new_in_graph)
            new_to_orig_in_inds = in_tree.query(new_in_graph, k=1)[1]
            orig_to_new_in_inds = new_kd_tree.query(original_in_graph, k=1)[1]
            orig_size = original_in_graph.shape[0]
            new_size = new_in_graph.shape[0]

            denominator = torch.bincount(
                torch.tensor(new_to_orig_in_inds, device=x.device), minlength=orig_size
            )
            denominator.requires_grad = False
            orig_pointing_elsewhere = np.arange(orig_size)
            orig_pointing_elsewhere = orig_pointing_elsewhere[
                orig_pointing_elsewhere != new_to_orig_in_inds[orig_to_new_in_inds]
            ]
            if orig_pointing_elsewhere.shape[0] > 0:
                index = torch.from_numpy(orig_pointing_elsewhere)
                values = torch.ones(
                    orig_pointing_elsewhere.shape[0], dtype=int, requires_grad=False
                )
                denominator = denominator.index_add_(0, index, values)
            scaled_weight = weight / denominator
            weight = scaled_weight[..., new_to_orig_in_inds]
            if orig_pointing_elsewhere.shape[0] > 0:
                index = torch.from_numpy(orig_to_new_in_inds[orig_pointing_elsewhere])
                values = scaled_weight[..., orig_pointing_elsewhere]
                weight = weight.index_add_(1, index, values)

        # -- DECODER-style --
        if new_out_graph is not None:
            new_kd_tree = KDTree(new_out_graph)
            new_to_orig_in_inds = out_tree.query(new_out_graph, k=1)[1]
            orig_to_new_in_inds = new_kd_tree.query(original_out_graph, k=1)[1]
            orig_size = original_out_graph.shape[0]
            new_size = new_out_graph.shape[0]

            new_weight = weight[new_to_orig_in_inds]
            new_bias = bias[new_to_orig_in_inds] if bias is not None else None
            denominator = torch.ones(
                new_size, device=x.device, dtype=int, requires_grad=False
            )
            orig_pointing_elsewhere = np.arange(orig_size)
            orig_pointing_elsewhere = orig_pointing_elsewhere[
                orig_pointing_elsewhere != new_to_orig_in_inds[orig_to_new_in_inds]
            ]
            if orig_pointing_elsewhere.shape[0] > 0:
                index = torch.from_numpy(orig_to_new_in_inds[orig_pointing_elsewhere])
                values = torch.ones(orig_pointing_elsewhere.shape[0], dtype=int)
                denominator = denominator.index_add_(0, index, values)
                values = weight[orig_pointing_elsewhere]
                new_weight = new_weight.index_add_(0, index, values)
                if bias is not None:
                    values = bias[orig_pointing_elsewhere]
                    new_bias = new_bias.index_add_(0, index, values)
                    new_bias = new_bias / denominator
                new_weight = new_weight / denominator.unsqueeze(1)
            weight = new_weight
            bias = new_bias

        return x @ weight.T + bias if bias is not None else x @ weight.T
