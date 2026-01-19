import pytest
from gfn.nn_lookup import NNLookupSciPy, NNLookupFaiss
import torch

try:
    import faiss  # noqa: F401

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def _setup_data():
    points = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    query = torch.tensor([[1.2, 2.7], [3.1, 4.2]])
    return points, query


def test_scipy_lookup():
    points, query = _setup_data()
    lookup = NNLookupSciPy(points)
    indices = lookup.query(query)
    expected_indices = torch.tensor([0, 1])
    torch.testing.assert_close(torch.tensor(indices), expected_indices)


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_faiss_lookup():
    points, query = _setup_data()
    lookup = NNLookupFaiss(points).to("cpu")
    indices = lookup.query(query)
    expected_indices = torch.tensor([0, 1])
    torch.testing.assert_close(torch.tensor(indices), expected_indices)
