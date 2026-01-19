User Guide
==========

.. .. toctree::
..    :titlesonly:
..    :maxdepth: 4

**gfn** defines one key class: ``gfn.GFNLayer``, which implements a Graph Feedforward Network layer.
Using **gfn** is intuitive - the ``GFN`` layer is an extension of the ``torch.nn.Linear`` layer.
Simply import it with ``from gfn import GFN`` and use it as outlined below.

If needed, see the :doc:`api` for further details.
GFN can be used in four different ways, depending on whether the inputs or outputs are vector-based or graph-based.

Graphical input and output
--------------------------

This is the most general case, where the layer operates on graphs at both input and output.

.. code-block:: python

   # initial input graph of 2 nodes
   original_in_graph = torch.tensor([(0,0), (1,0)])
   # initial output graph of 3 nodes
   original_out_graph = torch.tensor([(0,0), (-1,0), [0,-1]])
   gfn_layer = GFN(in_features=original_in_graph, out_features=original_out_graph)

   # new input graph of 4 nodes
   new_in_graph = torch.tensor([(0.5, 0.5), (1.5, 1.5), (1, 1), (0, 0.5)])
   # new output graph of 5 nodes
   new_out_graph = torch.tensor([(-0.5, -0.5), (-1.5, -1.5), (-1, -1), (0, -0.5), (-0.5, 0)])
   # input features for the new input graph (with 4 nodes)
   x = torch.ones(4)

   # predict for the new output graph using the new input graph
   y = gfn_layer(x, out_graph=new_out_graph)
   assert y.shape[-1] == 5 # (5 nodes in new output graph)

Graphical input only, vector output
-----------------------------------

The input is graphical, while the output is vector-based. This is therefore a graph pooling layer, or can also be thought of as an encoder-style layer.

.. code-block:: python

   # initial input graph with 2 nodes at (0, 0) and (1, 0)
   original_in_graph = torch.tensor([(0,0), (1,0)])
   # output is a constant vector with 3 features
   out_vector_size = 3
   gfn_layer = GFN(in_features=original_in_graph, out_features=out_vector_size)

   # new input graph of 4 nodes: (0.5, 0.5), (1.5, 1.5), (1,1) and (0, 0.5)
   new_in_graph = torch.tensor([(0.5, 0.5), (1.5, 1.5), (1, 1), (0, 0.5)])
   # input features for the new input graph (with 4 nodes)
   x = torch.ones(4)

   # predict the vector output using the new input graph
   y = gfn_layer(x, in_graph=new_in_graph)
   assert y.shape[-1] == 3 # (vector output of constant size 3)

Graphical output only, vector input
-----------------------------------

The input is vector-based, while the output is graphical. This is therefore a graph unpooling layer, or can also be thought of as a decoder-style layer.

.. code-block:: python

   # input is a constant vector with 2 features
   in_vector_size = 2
   # initial output graph with 3 nodes at (0, 0), (-1, 0) and (0, -1)
   original_out_graph = torch.tensor([(0,0), (-1,0), [0,-1]])
   gfn_layer = GFN(in_features=in_vector_size, out_features=original_out_graph)

   # new output graph of 5 nodes: (-0.5, -0.5), (-1.5, -1.5), (-1, -1), (0, -0.5) and (-0.5, 0)
   new_out_graph = torch.tensor([(-0.5, -0.5), (-1.5, -1.5), (-1, -1), (0, -0.5), (-0.5, 0)])
   # input features for the input vector (of constant size 2)
   x = torch.ones(2)

   # predict for the new output graph using the input vector
   y = gfn_layer(x, out_graph=new_out_graph)
   assert y.shape[-1] == 5 # (5 nodes in new output graph)

No graphs, vector input and output
----------------------------------

This case is trivially equivalent to a standard feedforward layer, and is included for completeness.
The implementation is identical to ``torch.nn.Linear``.

.. code-block:: python

   # input is a constant vector with 2 features
   in_vector_size = 2
   # output is a constant vector with 3 features
   out_vector_size = 3
   gfn_layer = GFN(in_features=in_vector_size, out_features=out_vector_size)

   # input features for the input vector (of constant size 2)
   x = torch.ones(2)

   y = gfn_layer(x)
   assert y.shape[-1] == 3 # (output vector of constant size 3)
