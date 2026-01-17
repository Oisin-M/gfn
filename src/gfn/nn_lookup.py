import torch
import numpy as np


def to_device(device):
    if isinstance(device, str):
        return torch.device(device)
    else:
        return device


def to_numpy(points):
    if isinstance(points, np.ndarray):
        return points
    elif isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    else:
        raise TypeError("Input points must be a numpy array or a torch tensor.")


class NNLookupSciPy:
    def __init__(self, points, device=None, dtype=None):
        from scipy.spatial import cKDTree as KDTree

        points = to_numpy(points)
        self.tree = KDTree(points)

    def query(self, points):
        points = to_numpy(points)
        # This returns numpy, but it is autoconverted to torch so not an issue
        _, indices = self.tree.query(points, k=1)
        return indices


class NNLookupFaiss(torch.nn.Module):
    def __init__(self, points, device=None, dtype=None):
        super().__init__()
        self.faiss = self.import_faiss()
        self.device = self._device(points) if device is None else to_device(device)
        self.gpu_support = self._has_gpu()
        self.index = self._build_index(points)
        self.res = None

    def _device(self, points):
        if isinstance(points, torch.Tensor):
            return points.device
        if isinstance(points, np.ndarray):
            return torch.device("cpu")
        raise TypeError("Input points must be a numpy array or a torch tensor.")

    def _has_gpu(self):
        return hasattr(self.faiss, "StandardGpuResources")

    def _build_index(self, points):
        points_np = to_numpy(points).astype("float32")
        index = self.faiss.IndexFlatL2(points_np.shape[1])

        if self.device.type == "cuda" and self.gpu_support:
            self.res = self.faiss.StandardGpuResources()
            index = self.faiss.index_cpu_to_gpu(self.res, 0, index)

        index.add(points_np)

        return index

    def query(self, points):
        if self.device.type == "cpu":
            points_np = to_numpy(points)
        else:
            points_np = to_numpy(points)
        _, indices = self.index.search(points_np, 1)
        # returns numpy, but it is autoconverted to torch so not an issue
        return indices.squeeze()

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, str):
            device = torch.device(device)

        if self.device.type == "cpu" and device.type == "cuda" and self.gpu_support:
            self.res = (
                self.faiss.StandardGpuResources() if self.res is None else self.res
            )
            self.index = self.faiss.index_cpu_to_gpu(self.res, 0, self.index)
            self.device = device
        elif self.device.type == "cuda" and device.type == "cpu":
            self.index = self.faiss.index_gpu_to_cpu(self.index)
            self.device = device
        else:
            pass

        return self

    def import_faiss(self):
        try:
            import faiss

            return faiss
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Faiss not found. Please install it to use Faiss-backed nearest neighbour lookup."
            )
