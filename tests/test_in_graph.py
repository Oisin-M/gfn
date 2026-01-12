import torch
import gfn
import pytest


def _setup_linear():
    weight = torch.tensor(
        [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]], dtype=float
    )
    bias = torch.tensor([1, 10, 100, 1000], dtype=float)

    linear_layer = torch.nn.Linear(3, 4, dtype=float)

    with torch.no_grad():
        linear_layer.weight.copy_(weight)
        linear_layer.bias.copy_(bias)

    return linear_layer


def _setup_gfn():
    weight = torch.tensor(
        [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]], dtype=float
    )
    bias = torch.tensor([1, 10, 100, 1000], dtype=float)

    in_graph = torch.tensor([(0.5, 0.5), (1.5, 1.5), (1, 1)])

    gfn_layer = gfn.GFN(in_graph, 4, dtype=float)

    with torch.no_grad():
        gfn_layer.weight.copy_(weight)
        gfn_layer.bias.copy_(bias)

    return gfn_layer


def test_matches_linear_no_graph():
    gfn_layer = _setup_gfn()
    linear_layer = _setup_linear()

    x = torch.tensor([0, 1, 2], dtype=float).reshape(1, -1)

    y_linear = linear_layer(x)
    y_gfn = gfn_layer(x)

    torch.testing.assert_close(y_gfn, y_linear)

    ((y_linear - 1) ** 2).sum().backward()
    ((y_gfn - 1) ** 2).sum().backward()

    assert gfn_layer.weight.grad is not None
    assert linear_layer.weight.grad is not None
    assert gfn_layer.bias.grad is not None
    assert linear_layer.bias.grad is not None

    torch.testing.assert_close(gfn_layer.weight.grad, linear_layer.weight.grad)
    torch.testing.assert_close(gfn_layer.bias.grad, linear_layer.bias.grad)


def test_matches_linear_same_graph():
    gfn_layer = _setup_gfn()
    linear_layer = _setup_linear()

    x = torch.tensor([0, 1, 2], dtype=float).reshape(1, -1)
    in_graph = torch.tensor([(0.5, 0.5), (1.5, 1.5), (1, 1)])

    y_linear = linear_layer(x)
    y_gfn = gfn_layer(x, in_graph=in_graph)

    torch.testing.assert_close(y_gfn, y_linear)

    ((y_linear - 1) ** 2).sum().backward()
    ((y_gfn - 1) ** 2).sum().backward()

    assert gfn_layer.weight.grad is not None
    assert linear_layer.weight.grad is not None
    assert gfn_layer.bias.grad is not None
    assert linear_layer.bias.grad is not None

    torch.testing.assert_close(gfn_layer.weight.grad, linear_layer.weight.grad)
    torch.testing.assert_close(gfn_layer.bias.grad, linear_layer.bias.grad)


def test_matches_linear_ones():
    gfn_layer = _setup_gfn()
    linear_layer = _setup_linear()

    in_graph = torch.tensor([(0, 0), (1.25, 1.5), (2, 2), (3, 3), (-1, 1)])

    x_linear = torch.tensor([1, 1, 1], dtype=float).reshape(1, -1)
    x_gfn = torch.ones(in_graph.shape[0], dtype=float).reshape(1, -1)

    y_linear = linear_layer(x_linear)
    y_gfn = gfn_layer(x_gfn, in_graph=in_graph)

    torch.testing.assert_close(y_gfn, y_linear)


def test_prediction():
    gfn_layer = _setup_gfn()
    x = torch.tensor([-100, -10, 10, 100], dtype=float).reshape(1, -1)

    in_graph = torch.tensor([(0, 0), (0.45, 0.45), (0.7, 0.7), (1.9, 1.9)])

    y_gfn = gfn_layer(x, in_graph=in_graph)

    weights_expected = torch.tensor(
        [
            [1 / 3, 1 / 3, 1 / 3 + 100, 10],
            [2 / 3, 2 / 3, 2 / 3 + 200, 20],
            [3 / 3, 3 / 3, 3 / 3 + 300, 30],
            [4 / 3, 4 / 3, 4 / 3 + 400, 40],
        ],
        dtype=float,
    )
    bias_expected = torch.tensor([1, 10, 100, 1000], dtype=float)
    expected = x @ weights_expected.T + bias_expected

    torch.testing.assert_close(y_gfn, expected)

    ((y_gfn - 1) ** 2).sum().backward()

    assert gfn_layer.weight.grad is not None
    assert gfn_layer.bias.grad is not None

    # TODO: double check these
    weights_gradient_expected = torch.tensor(
        [
            [-131111.1111, 393333.3333, 39333.3333],
            [-262822.2222, 788466.6667, 78846.6667],
            [-399933.3333, 1199800.0000, 119980.0000],
            [-591044.4444, 1773133.3333, 177313.3333],
        ],
        dtype=float,
    )
    bias_gradient_expected = torch.tensor(
        [3933.3333, 7884.6667, 11998.0000, 17731.3333], dtype=float
    )

    torch.testing.assert_close(gfn_layer.weight.grad, weights_gradient_expected)
    torch.testing.assert_close(gfn_layer.bias.grad, bias_gradient_expected)


def test_raises():
    gfn_layer = _setup_gfn()

    x = torch.tensor([0, 1, 2], dtype=float).reshape(1, -1)
    in_graph = torch.tensor([(0.5, 0.5), (1.5, 1.5), (1, 1)])

    with pytest.raises(ValueError):
        gfn_layer(x, in_graph=in_graph, out_graph=in_graph)
