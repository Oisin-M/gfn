import torch
import gfn


def _setup():
    weight = torch.tensor(
        [[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]], dtype=float
    )
    bias = torch.tensor([1, 10, 100, 1000], dtype=float)

    linear_layer = torch.nn.Linear(3, 4, dtype=float)
    gfn_layer = gfn.GFN(3, 4, dtype=float)

    with torch.no_grad():
        linear_layer.weight.copy_(weight)
        gfn_layer.weight.copy_(weight)
        linear_layer.bias.copy_(bias)
        gfn_layer.bias.copy_(bias)

    return linear_layer, gfn_layer


def test_matches_linear():
    linear_layer, gfn_layer = _setup()

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
