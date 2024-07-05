import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from sklearn.preprocessing import RobustScaler


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))

        # メモリ効率の良いロバストスケーリングの適用
        X_shape = self.X.shape
        scaler = RobustScaler()
        
        # データを小さなバッチに分割して処理
        batch_size = 1000  # この値は調整が必要かもしれません
        scaled_data = []
        
        for i in range(0, X_shape[0], batch_size):
            batch = self.X[i:i+batch_size].numpy().astype(np.float32)  # float32を使用
            batch_reshaped = batch.reshape(-1, X_shape[-1])
            batch_scaled = scaler.fit_transform(batch_reshaped)
            scaled_data.append(batch_scaled.reshape(-1, X_shape[1], X_shape[2]))
        
        self.X = torch.tensor(np.concatenate(scaled_data), dtype=torch.float32)

        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]