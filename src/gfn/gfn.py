import torch
import numpy as np
from gfn.nn_lookup import NNLookupSciPy, NNLookupFaiss


class GFN(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        nn_backend="scipy",
    ):
        if type(in_features) is int and type(out_features) is int:
            print(
                "Warning: no graphical data provided to GFN layer. Behaves like a standard Linear layer."
            )

        if nn_backend == "scipy":
            self.lookup = NNLookupSciPy
        elif nn_backend == "faiss":
            self.lookup = NNLookupFaiss
        else:
            raise ValueError(f"Unknown nn_backend: {nn_backend}")

        in_features_size = (
            in_features.shape[0] if type(in_features) is not int else in_features
        )
        out_features_size = (
            out_features.shape[0] if type(out_features) is not int else out_features
        )

        super().__init__(in_features_size, out_features_size, bias, device, dtype)

        if type(in_features) is not int:
            self.in_tree = self.lookup(in_features, device=device, dtype=dtype)
            self.in_graph = in_features
        else:
            self.in_tree = None
            self.in_graph = None

        if type(out_features) is not int:
            self.out_tree = self.lookup(out_features, device=device, dtype=dtype)
            self.out_graph = out_features
        else:
            self.out_tree = None
            self.out_graph = None

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

        device = x.device

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
            with torch.no_grad():
                new_kd_tree = self.lookup(new_in_graph)
                new_to_orig_in_inds = in_tree.query(new_in_graph)
                orig_to_new_in_inds = new_kd_tree.query(original_in_graph)
                orig_size = original_in_graph.shape[0]
                new_size = new_in_graph.shape[0]

                denominator = torch.bincount(
                    torch.tensor(new_to_orig_in_inds, device=device),
                    minlength=orig_size,
                ).to(device)
                denominator.requires_grad = False
                orig_pointing_elsewhere = np.arange(orig_size)
                orig_pointing_elsewhere = orig_pointing_elsewhere[
                    orig_pointing_elsewhere != new_to_orig_in_inds[orig_to_new_in_inds]
                ]

            if orig_pointing_elsewhere.shape[0] > 0:
                with torch.no_grad():
                    index = torch.as_tensor(orig_pointing_elsewhere, device=device)
                    values = torch.ones(
                        orig_pointing_elsewhere.shape[0],
                        dtype=int,
                        requires_grad=False,
                        device=device,
                    )
                    denominator = denominator.index_add_(0, index, values)
            scaled_weight = weight / denominator
            weight = scaled_weight[..., new_to_orig_in_inds]
            if orig_pointing_elsewhere.shape[0] > 0:
                index = torch.as_tensor(
                    orig_to_new_in_inds[orig_pointing_elsewhere], device=device
                )
                values = scaled_weight[..., orig_pointing_elsewhere]
                weight = weight.index_add_(1, index, values)

        # -- DECODER-style --
        if new_out_graph is not None:
            with torch.no_grad():
                new_kd_tree = self.lookup(new_out_graph)
                new_to_orig_in_inds = out_tree.query(new_out_graph)
                orig_to_new_in_inds = new_kd_tree.query(original_out_graph)
                orig_size = original_out_graph.shape[0]
                new_size = new_out_graph.shape[0]

                denominator = torch.ones(
                    new_size, device=device, dtype=int, requires_grad=False
                )
                orig_pointing_elsewhere = np.arange(orig_size)
                orig_pointing_elsewhere = orig_pointing_elsewhere[
                    orig_pointing_elsewhere != new_to_orig_in_inds[orig_to_new_in_inds]
                ]
            new_weight = weight[new_to_orig_in_inds]
            new_bias = bias[new_to_orig_in_inds] if bias is not None else None
            if orig_pointing_elsewhere.shape[0] > 0:
                with torch.no_grad():
                    index = torch.as_tensor(
                        orig_to_new_in_inds[orig_pointing_elsewhere], device=device
                    )
                    values = torch.ones(
                        orig_pointing_elsewhere.shape[0],
                        dtype=int,
                        requires_grad=False,
                        device=device,
                    )
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
