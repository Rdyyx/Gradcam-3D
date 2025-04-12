import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage import zoom

class_map = {'data': 0}

class Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []

        for label in class_map.keys():
            class_path = os.path.join(data_path, label)
            for filename in os.listdir(class_path):
                if filename.endswith('.nii'):
                    file_path = os.path.join(class_path, filename)
                    self.samples.append((file_path, class_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        nii_image = nib.load(file_path)
        data = nii_image.get_fdata()

        # 调整尺寸为64x64x64
        if data.shape != (64, 64, 64):
            data = self.resize_data(data, (64, 64, 64))

        # 添加通道维度并归一化
        data = np.expand_dims(data, axis=0)
        data = data.astype(np.float32)

        # 使用Z-score标准化
        mean = np.mean(data)
        std = np.std(data)
        if std != 0:
            data = (data - mean) / std

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def resize_data(data, target_shape):
        if data.ndim == 4:
            data = data[..., 0]
        factors = [target_shape[i] / data.shape[i] for i in range(3)]
        return zoom(data, factors, order=1)