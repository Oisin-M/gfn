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

    out_graph = torch.tensor([(-3, 2), (1, 0), (2, 1), (3, 2)])

    gfn_layer = gfn.GFN(3, out_graph, dtype=float)

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
    out_graph = torch.tensor([(-3, 2), (1, 0), (2, 1), (3, 2)])

    y_linear = linear_layer(x)
    y_gfn = gfn_layer(x, out_graph=out_graph)

    torch.testing.assert_close(y_gfn, y_linear)

    ((y_linear - 1) ** 2).sum().backward()
    ((y_gfn - 1) ** 2).sum().backward()

    assert gfn_layer.weight.grad is not None
    assert linear_layer.weight.grad is not None
    assert gfn_layer.bias.grad is not None
    assert linear_layer.bias.grad is not None

    torch.testing.assert_close(gfn_layer.weight.grad, linear_layer.weight.grad)
    torch.testing.assert_close(gfn_layer.bias.grad, linear_layer.bias.grad)


def test_prediction():
    gfn_layer = _setup_gfn()
    x = torch.tensor([-100, -10, 10], dtype=float).reshape(1, -1)

    out_graph = torch.tensor([(-5, -5), (-10, -10), (0, 0), (3, 1.5), (3, 3)])

    y_gfn = gfn_layer(x, out_graph=out_graph)

    out_graph = torch.tensor([(-3, 2), (1, 0), (2, 1), (3, 2)])

    weights_expected = torch.tensor(
        [
            [1, 10, 100],
            [1, 10, 100],
            [(1 + 2) / 2, (10 + 20) / 2, (100 + 200) / 2],
            [(3 + 4) / 2, (30 + 40) / 2, (300 + 400) / 2],
            [4, 40, 400],
        ],
        dtype=float,
    )
    bias_expected = torch.tensor(
        [1, 1, (1 + 10) / 2, (100 + 1000) / 2, 1000], dtype=float
    )
    expected = x @ weights_expected.T + bias_expected

    torch.testing.assert_close(y_gfn, expected)

    ((y_gfn - 1) ** 2).sum().backward()

    assert gfn_layer.weight.grad is not None
    assert gfn_layer.bias.grad is not None

    # TODO: double check these
    weights_gradient_expected = torch.tensor(
        [
            [-440450.0, -44045.0, 44045.0],
            [-120450.0, -12045.0, 12045.0],
            [-334900.0, -33490.0, 33490.0],
            [-1174700.0, -117470.0, 117470.0],
        ],
        dtype=float,
    )
    bias_gradient_expected = torch.tensor(
        [4404.5000, 1204.5000, 3349.0000, 11747.0000], dtype=float
    )

    torch.testing.assert_close(gfn_layer.weight.grad, weights_gradient_expected)
    torch.testing.assert_close(gfn_layer.bias.grad, bias_gradient_expected)


def test_raises():
    gfn_layer = _setup_gfn()

    x = torch.tensor([0, 1, 2], dtype=float).reshape(1, -1)
    out_graph = torch.tensor([(-3, 2), (1, 0), (2, 1), (3, 2)])

    with pytest.raises(ValueError):
        gfn_layer(x, in_graph=out_graph, out_graph=out_graph)
