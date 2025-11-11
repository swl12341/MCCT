# dataset.py
from torch.utils.data import Dataset, TensorDataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torch

class IMUDataset(Dataset):
    def __init__(self, mat_file, load_extra=False):
        data = loadmat(mat_file)
        self.X = torch.tensor(data['X_tensor'], dtype=torch.float32)
        self.y = torch.tensor(data['y_tensor'].squeeze(), dtype=torch.long)

        # ✅ 加载额外信息（仅用于测试集）
        self.extra = {}
        if load_extra:
            for key in data:
                if key not in ['__header__', '__version__', '__globals__', 'X_tensor', 'y_tensor']:
                    self.extra[key] = data[key]  # 原始 NumPy 格式

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def stratified_split(dataset, test_size=0.2):
    X = dataset.X.numpy()
    y = dataset.y.numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    return train_dataset, val_dataset
