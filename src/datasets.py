import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


class ThingsMEGDataset(torch.utils.data.Dataset):
    @staticmethod
    def compute_parameters(data_dir):
        print("Computing parameters from training data...")
        X = torch.load(os.path.join(data_dir, "train_X.pt"))
        X_shape = X.shape
        batch_size = 1000

        # 平均と標準偏差の計算
        mean_sum = np.zeros(X_shape[-1], dtype=np.float32)
        squared_sum = np.zeros(X_shape[-1], dtype=np.float32)
        count = 0
        for i in tqdm(range(0, X_shape[0], batch_size), desc="Calculating mean and std"):
            batch = X[i:i+batch_size].numpy().astype(np.float32)
            batch_reshaped = batch.reshape(-1, X_shape[-1])
            mean_sum += np.sum(batch_reshaped, axis=0)
            squared_sum += np.sum(np.square(batch_reshaped), axis=0)
            count += batch_reshaped.shape[0]
        
        mean = mean_sum / count
        std = np.sqrt(squared_sum / count - np.square(mean))
        
        # クリッピングの閾値
        clip_min = mean - 20 * std
        clip_max = mean + 20 * std

        # クリッピングとデータの収集
        clipped_data = []
        for i in tqdm(range(0, X_shape[0], batch_size), desc="Clipping data"):
            batch = X[i:i+batch_size].numpy().astype(np.float32)
            batch_reshaped = batch.reshape(-1, X_shape[-1])
            batch_clipped = np.clip(batch_reshaped, clip_min, clip_max)
            clipped_data.append(batch_clipped)

        # RobustScalerのフィッティング
        print("Fitting RobustScaler...")
        scaler = RobustScaler()
        scaler.fit(np.vstack(clipped_data))

        # パラメータの保存
        params = {
            'clip_min': clip_min,
            'clip_max': clip_max,
            'scaler': scaler
        }
        joblib.dump(params, os.path.join(data_dir, 'preprocessing_params.joblib'))
        print("Parameters computed and saved.")

    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        # パラメータのロード（存在しない場合は計算）
        params_file = os.path.join(data_dir, 'preprocessing_params.joblib')
        if not os.path.exists(params_file):
            self.compute_parameters(data_dir)
        params = joblib.load(params_file)
        clip_min, clip_max, scaler = params['clip_min'], params['clip_max'], params['scaler']

        # データの読み込みと処理
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        X_shape = self.X.shape
        batch_size = 1000

        scaled_data = []
        for i in tqdm(range(0, X_shape[0], batch_size), desc=f"Processing {split} data"):
            batch = self.X[i:i+batch_size].numpy().astype(np.float32)
            batch_reshaped = batch.reshape(-1, X_shape[-1])
            
            # クリッピングの適用
            batch_clipped = np.clip(batch_reshaped, clip_min, clip_max)
            
            # スケーリングの適用
            batch_scaled = scaler.transform(batch_clipped)
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