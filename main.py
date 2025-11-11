# main.py
import os
import torch
import random
import numpy as np
from scipy.io import savemat
from torch.utils.data import DataLoader
from config import *
# BiLSTMClassifier    LSTMClassifier
from model2 import CNNTransformerClassifier, LSTMClassifier, BiLSTMClassifier
from dataset import IMUDataset, stratified_split
from train import train_model
from visualize import plot_training_curves, plot_attention_weights, plot_tsne_with_misclass

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    print("Using device:", device)

    base_dir = r'output'
    prefix = 'run_'
    i = 1
    while os.path.exists(os.path.join(base_dir, f'{prefix}{i}')):
        i += 1
    output_dir = os.path.join(base_dir, f'{prefix}{i}')
    os.makedirs(output_dir)

    print(f"[INFO] 本次结果将保存在: {output_dir}")

    dataset = IMUDataset(mat_path)
    train_dataset, val_dataset = stratified_split(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 对比试验
    # BiLSTMClassifier    LSTMClassifier CNNTransformerClassifier
    # model = LSTMClassifier(
    #     input_dim=input_dim,
    #     seq_len=seq_len,
    #     num_classes=2,
    #     hidden_dim=128,
    #     num_layers=2,
    #     dropout=dropout,  # 确保 dropout 的值在 [0, 1] 之间
    #     d_model=d_model,  # 这些参数不会被使用
    #     nhead=nhead  # 这些参数也不会被使用
    # )
    print(input_dim, seq_len, num_classes, d_model, nhead, num_layers, dropout)
    model = CNNTransformerClassifier(input_dim, seq_len, num_classes, d_model, nhead, num_layers, dropout)
    # 训练并获得损失、准确率和AUC
    train_losses, val_accuracies, val_aucs, best_epoch = train_model(
        model, train_loader, val_loader, device, num_epochs, output_dir, dataset
    )

    # 可视化训练曲线
    plot_training_curves(train_losses, val_accuracies, val_aucs, output_dir)
    sample_X, _ = val_dataset[0]
    plot_attention_weights(model, sample_X, device, output_dir)
    plot_tsne_with_misclass(model, val_loader, device, output_dir)

    # 保存 train_losses 为 .mat 文件
    mat_filename = os.path.join(output_dir, "train_losses.mat")
    savemat(mat_filename, {"train_losses": np.array(train_losses)})
    print(f"[INFO] train_losses 已保存为 {mat_filename}")

    print(f"[INFO] 最好的模型是第 {best_epoch} 个 epoch")