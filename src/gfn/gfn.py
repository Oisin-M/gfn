import torch
import numpy as np
from gfn.nn_lookup import NNLookupSciPy, NNLookupFaiss


class GFN(torch.nn.Linear):
    r"""The graph feedforward network (GFN) layer from `"GFN: A graph feedforward network for resolution-invariant reduced operator learning in multifidelity applications" <https://doi.org/10.1016/j.cma.2024.117458>`_.

        The layer is an extension of the standard :class:`torch.nn.Linear`, but with weights
        and biases are optionally associated to original input :math:`\mathcal{M}^i_o` and output :math:`\mathcal{M}^o_o` graphs.
        GFN then defines new weights and biases for new input :math:`\mathcal{M}^i_n` and output :math:`\mathcal{M}^o_n` graphs,
        allowing for resolution-invariant learning.

        The general GFN equation is given by:

        .. math::
            :nowrap:

            \begin{equation*}
            \begin{aligned}
            \tilde{W}_{i_{{\mathcal{M}^o_{n}}}j_{{\mathcal{M}^i_{n}}}} &= \underset{\forall l_{\mathcal{M}^o_{o}} \text{ s.t } l_{\mathcal{M}^o_{o}} {\leftarrow}\!{\backslash}\!{\rightarrow} i_{\mathcal{M}^o_{n}}}{\operatorname{mean}}
            \sum_{\forall k_{\mathcal{M}^i_{o}} \text{ s.t } k_{\mathcal{M}^i_{o}} {\leftarrow}\!{\backslash}\!{\rightarrow} j_{\mathcal{M}^i_{n}}} \frac{W_{l_{\mathcal{M}^o_{o}}k_{\mathcal{M}^i_{o}}}}{\lvert \{ h_{\mathcal{M}^i_{n}} \text{ s.t. } k_{\mathcal{M}^i_{o}} ~{\leftarrow}\!{\backslash}\!{\rightarrow}~ h_{\mathcal{M}^i_{n}} \}\rvert}, \\

            \tilde{b}^d_{i_{\mathcal{M}^o_{n}}} &=  \underset{\forall k_{\mathcal{M}^o_{o}} \text{ s.t } k_{\mathcal{M}^o_{o}} {\leftarrow}\!{\backslash}\!{\rightarrow} i_{\mathcal{M}^o_{n}}}{\operatorname{mean}} {b}^d_{k_{\mathcal{M}^o_{o}}}.
            \end{aligned}
            \end{equation*}

        where:

        - :math:`\mathcal{M}^i_{o}` is the original input graph,
        - :math:`\mathcal{M}^o_{o}` is the original output graph,
        - :math:`\mathcal{M}^i_{n}` is the new input graph,
        - :math:`\mathcal{M}^o_{n}` is the new output graph,
        - :math:`W` and :math:`b` are the weights and biases associated to the original graphs,
        - :math:`\tilde{W}` and :math:`\tilde{b}` are the new weights and biases associated to the new graphs,
        - :math:`i_{\mathcal{M}_1} {\leftarrow}\!{\backslash}\!{\rightarrow} j_{\mathcal{M}_2}` indicates that either node :math:`i` in graph :math:`\mathcal{M}_1` is the nearest neighbor of node :math:`j` in graph :math:`\mathcal{M}_2` or vice versa.

        GFN also supports mapping from graphs to vectors or vectors to graphs, in which case the above equation can be applied
        with :math:`\mathcal{M}^i_{o}=\mathcal{M}^i_{n}` for a vector input or :math:`\mathcal{M}^o_{o}=\mathcal{M}^o_{n}` for a vector output.

        Args:
            in_features (int or torch.Tensor): Either a tensor of shape :math:`(N_{\text{in}}, D_{\text{in}})` containing the coordinates of each original input node (graph input) or the size :math:`N_{\text{in}}` of each input (vector input).
            out_features (int or torch.Tensor): Either a tensor of shape :math:`(N_{\text{out}}, D_{\text{out}})` containing the coordinates of each original output node (graph output) or the size :math:`N_{\text{out}}` of each output (vector output).
            bias (bool, optional): If set to :obj:`False`, the layer will not learn
                an additive bias. (default: :obj:`True`)
            device (torch.device, optional): The device on which the layer should be
                allocated. (default: :obj:`None`)
            dtype (torch.dtype, optional): The data type for the layer's parameters.
                (default: :obj:`None`)
            nn_backend (str, optional): The backend to use for nearest neighbor
                lookup. Can be either :obj:`"scipy"` or :obj:`"faiss"`.
                (default: :obj:`"scipy"`)
    """

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
        r"""
        Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor of shape :math:`(..., N_{\text{in}}^{\prime})`.
            in_graph (torch.Tensor, optional): The input graph matching the shape of the input tensor :math:`(N_{\text{in}}^{\prime}, D_{\text{in}})`. If :obj:`None`, treats as vector input i.e. assumes no change from the original input graph. (default: :obj:`None`)
            out_graph (torch.Tensor, optional): The output graph of shape :math:`(N_{\text{out}}^{\prime}, D_{\text{out}})`. If :obj:`None`, treats as vector output i.e. assumes no change to the original output graph. (default: :obj:`None`)

        Returns:
            torch.Tensor: The output tensor. Matches the shape of the new output graph if provided.
        """
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
