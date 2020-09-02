import numpy as np
from torch.utils.data import Dataset


class MovingMnistDataset(Dataset):
    def __init__(self, path="./mnist_test_seq.npy", phase_train=True):
        self.data = np.load(path)
        # (t, N, H, W) -> (N, t, C, H, W)
        self.data = self.data.transpose(1, 0, 2, 3)[:, :, None, ...]
        if phase_train:
            self.data = self.data[:9000]
        else:
            self.data = self.data[9000:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (self.data[i, :10, ...] / 255).astype(np.float32), (self.data[i, 10:, ...]/255).astype(np.float32)
