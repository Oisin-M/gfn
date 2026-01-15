import torch
import numpy as np

def to_numpy(points):
    if isinstance(points, np.ndarray):
        return points
    elif isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    else:
        raise TypeError("Input points must be a numpy array or a torch tensor.")

class NNLookupSciPy:
    def __init__(self, points):
        from scipy.spatial import cKDTree as KDTree
        points = to_numpy(points)
        self.tree = KDTree(points)

    def query(self, points):
        points = to_numpy(points)
        # This returns numpy, but it is autoconverted to torch so not an issue
        _, indices = self.tree.query(points, k=1)
        return indices
    
class NNLookupFAISS:
    def __init__(self, points):
        self.device = self._device(points)
        self.gpu_support = self._has_gpu()
        self.index = self._build_index(points)

    def _device(self, points):
        if isinstance(points, torch.Tensor):
            return points.device
        if isinstance(points, np.ndarray):
            return torch.device('cpu')
        raise TypeError("Input points must be a numpy array or a torch tensor.")

    def _has_gpu(self):
        import faiss
        return hasattr(faiss, "StandardGpuResources")

    def _build_index(self, points):
        import faiss
        points_np = to_numpy(points).astype('float32')
        index = faiss.IndexFlatL2(points_np.shape[1])
        index.add(points_np)

        if self.device.type == "cuda" and self.gpu_support:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        return index

    def query(self, points):
        if self.device.type == "cpu":
            points_np = to_numpy(points).astype('float32')
        else:
            points_np = to_numpy(points).astype('float32')  # FAISS GPU still expects numpy
        _, indices = self.index.search(points_np, 1)
        # returns numpy, but it is autoconverted to torch so not an issue
        return indices.squeeze()

    def to(self, device):
        import faiss
        if self.device.type == 'cpu' and device.type == 'cuda' and self.gpu_support:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        elif self.device.type == 'cuda' and device.type == 'cpu':
            self.index = faiss.index_gpu_to_cpu(self.index)

        self.device = device