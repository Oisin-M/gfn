# Graph Feedforward Network (GFN) - a novel neural network layer for resolution-invariant machine learning

<p align="center">
  <a href="https://doi.org/10.1016/j.cma.2024.117458">
    <img src="https://badgen.net/badge/10.1016/j.cma.2024.117458/red?icon=github/" alt="doi">
  </a>
  <!-- <a href="https://codecov.io/gh/Oisin-M/gfn">
    <img src="https://codecov.io/gh/Oisin-M/gfn/branch/develop/graph/badge.svg" alt="Code Coverage">
  </a> -->
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
  <a href="https://github.com/Oisin-M/gfn/releases">
    <img src="https://img.shields.io/github/v/release/Oisin-M/gfn?color=purple&label=Release" alt="Latest Release">
  </a>
</p>

<p align="center">
  <a href="#why-gfns">Why GFNs?</a>
  •
  <a href="https://gfn-layer.readthedocs.io">Documentation</a>
  •
  <a href="#installation">Installation</a>
  •
  <a href="#quickstart">Quickstart</a>
  •
  <a href="#citing">Citing</a>
</p>

GFN is a generalisation of feedforward networks for graphical data.

> [!IMPORTANT]
> The code reproducing the results in the GFN-ROM paper has now been moved to [Oisin-M/GFN-ROM](https://github.com/Oisin-M/GFN-ROM).

## Why GFNs?
Many applications rely upon graphical data, which standard machine learning methods such as feedforward networks and convolutions cannot handle. GFNs present a novel approach of tackling this problem by extending existing machine learning approaches for use on graphical data. GFNs have very close links with neural operators and graph neural networks.

<p align="center">
    <img src="https://github.com/Oisin-M/gfn/raw/refs/heads/main/docs/images/gfn.png"/>
</p>

Key advantages of GFNs:
- Resolution invariance
- Equivalence to feedforward networks for single fidelity data (no deterioration in performance)
- Provable guarantees on performance for super- and sub-resolution
- Both fixed and adapative multifidelity training possible

## Installation

`gfn` is readily available on PyPI.
```
pip install gfn-layer
```
**Note:** the package name on PyPI is **gfn-layer**, gfn refers to a different package.

For a developer installation
```
git clone https://github.com/Oisin-M/gfn.git
cd gfn
pip install -e .
pre-commit install
```

## Quickstart

Using `gfn` is intuitive - the `GFN` layer is an extension of the `torch.nn.Linear` layer. Simply import it with `from gfn import GFN` and use as follows:

**No graph (equivalent to `torch.nn.Linear`)**
```python
gfn_layer = GFN(in_features=2, out_features=3)

x = torch.ones(2)

y = gfn_layer(x)
assert y.shape[-1] == 3
```

**In graph only**
```python
# in graph with 2 nodes at (0, 0) and (1, 0)
original_in_graph = torch.tensor([(0,0), (1,0)])
gfn_layer = GFN(in_features=original_in_graph, out_features=3)

# predict using graph of 4 nodes: (0.5, 0.5), (1.5, 1.5), (1,1) and (0, 0.5)
new_in_graph = torch.tensor([(0.5, 0.5), (1.5, 1.5), (1, 1), (0, 0.5)])
x = torch.ones(4)

y = gfn_layer(x, in_graph=new_in_graph)
assert y.shape[-1] == 3
```

**Out graph only**
```python
# out graph with 3 nodes at (0, 0), (-1, 0) and (0, -1)
original_out_graph = torch.tensor([(0,0), (-1,0), [0,-1]])
gfn_layer = GFN(in_features=2, out_features=original_out_graph)

# predict at graph of 5 nodes: (-0.5, -0.5), (-1.5, -1.5), (-1, -1), (0, -0.5) and (-0.5, 0)
new_out_graph = torch.tensor([(-0.5, -0.5), (-1.5, -1.5), (-1, -1), (0, -0.5), (-0.5, 0)])
x = torch.ones(2)

y = gfn_layer(x, out_graph=new_out_graph)
assert y.shape[-1] == 5
```

**Both out graph and in graph**
```python
# in graph of 2 nodes
original_in_graph = torch.tensor([(0,0), (1,0)])
# out graph of 3 nodes
original_out_graph = torch.tensor([(0,0), (-1,0), [0,-1]])
gfn_layer = GFN(in_features=original_in_graph, out_features=original_out_graph)

# predict using graph of 4 nodes
new_in_graph = torch.tensor([(0.5, 0.5), (1.5, 1.5), (1, 1), (0, 0.5)])
# predict at graph of 5 nodes
new_out_graph = torch.tensor([(-0.5, -0.5), (-1.5, -1.5), (-1, -1), (0, -0.5), (-0.5, 0)])
x = torch.ones(4)

y = gfn_layer(x, out_graph=new_out_graph)
assert y.shape[-1] == 5
```

## Citing
If this work is useful to you, please cite

[1] Morrison, O. M., Pichi, F. and Hesthaven, J. S. (2024) ‘GFN: A graph feedforward network for resolution-invariant reduced operator learning in multifidelity applications’. Available at: [arXiv](https://arxiv.org/abs/2406.03569) and [Computer Methods in Applied Mechanics and Engineering](https://doi.org/10.1016/j.cma.2024.117458)
```
@article{Morrison2024,
  title = {{GFN}: {A} graph feedforward network for resolution-invariant reduced operator learning in multifidelity applications},
  author = {Morrison, Oisín M. and Pichi, Federico and Hesthaven, Jan S.},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume = {432},
  pages = {117458},
  year = {2024},
  doi = {10.1016/j.cma.2024.117458},
}
```
