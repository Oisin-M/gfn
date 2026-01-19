Motivation
==========

.. .. toctree::
..    :titlesonly:
..    :maxdepth: 4

The feedforward network is the fundamental building block of deep learning models, powering everything from image classification to natural language processing.
However, standard feedforward networks are designed to operate on fixed-size vector inputs and outputs, limiting their applicability to data with changing structures and sizes.

As an illustrative example, consider the common task of an autoencoder for graphical inputs. A natural approach is to use a standard feedforward network by flattening the graphical input into a vector, compressing the vector through the network, and then decompressing and reshaping the output back into a graph.

.. image:: ../images/ae.png
   :alt: A schematic of a feedforward network-based autoencoder.
   :align: center

This approach can give state-of-the-art results when the training and testing data are of the same resolution, for example the GCA-ROM :cite:`gcarom` approach for model order reduction.
However, because standard feedforward networks operate on fixed-size vectors, this approach breaks down when any of the training or testing data are of different resolutions.

Nonetheless, suppose that we have a trained autoencoder as above performing well on a fixed graph. Now, we wish to use this trained model to make predictions on a different but similar graph.
It is intuitively clear that one should be able to leverage the information from the previous model to make a prediction on the new graph.

This is precisely the idea behind Graph Feedforward Networks (GFNs) :cite:`gfn`. The method defines a means of transferring weights and biases learned on one graph to another graph, enabling resolution-invariant learning or prediction. A key advantage is that GFNs have provable guarantees on performance when transferring between graphs.

.. image:: ../images/gfn.png
   :alt: A schematic of a GFN-based autoencoder.
   :align: center

Therefore, the original autoencoder example can be re-imagined using GFNs, where the encoder and decoder are now GFN layers. This extends the applicability of the autoencoder to inputs and outputs of varying resolutions.
GFN-ROM is an example of this approach for model order reduction.
