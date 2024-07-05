import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
from scipy import signal


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", compute_params: bool = False) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

        self.target_freq = 120 # 目標のサンプリング周波数
        self.original_freq = 1200  # 元のサンプリング周波数

        self.baseline_start = 0 # ベースラインの開始時刻
        self.baseline_end = 60 # ベースラインの終了時刻

        params_path = os.path.join(data_dir, "preprocessing_params.npy")
        if compute_params and split == "train":
            self.compute_and_save_params(params_path)
        self.load_params(params_path)

    def compute_and_save_params(self, params_path):
        print("Computing preprocessing parameters...")
        all_baselines = []
        for i in range(self.num_samples):
            X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
            X = np.load(X_path)
            X_downsampled = signal.resample(X, int(X.shape[1] * self.target_freq / self.original_freq), axis=1)
            all_baselines.append(X_downsampled[:, self.baseline_start:self.baseline_end])
        
        all_baselines = np.concatenate(all_baselines, axis=1)
        baseline_mean = np.mean(all_baselines, axis=1)
        baseline_std = np.std(all_baselines, axis=1)
        
        np.save(params_path, {"baseline_mean": baseline_mean, "baseline_std": baseline_std})
        print("Parameters saved.")

    def load_params(self, params_path):
        params = np.load(params_path, allow_pickle=True).item()
        self.baseline_mean = params["baseline_mean"]
        self.baseline_std = params["baseline_std"]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path)

        # ダウンサンプリングの適用
        X_downsampled = signal.resample(X, int(X.shape[1] * self.target_freq / self.original_freq), axis=1)
        
        # ベースライン補正の適用
        X_corrected = self.apply_baseline_correction(X_downsampled)
        
        X = torch.from_numpy(X_corrected)
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx

    def apply_baseline_correction(self, X):
        return X - self.baseline_mean[:, np.newaxis]

    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return int(np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1] * self.target_freq / self.original_freq)