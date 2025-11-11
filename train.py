# train.py

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from evaluate import evaluate_model
import os
from config import *

def train_model(model, train_loader, val_loader, device, num_epochs, output_dir, full_dataset):
    model.to(device)

    # 计算类别权重（仅用于训练时的损失函数）
    y_all = full_dataset.y.numpy()
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_all),
                                         y=y_all)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights[1])  # 仅对正类加权

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_epoch = 0
    train_losses, val_accuracies, val_aucs = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(X_batch)            # outputs: [B]
            loss = criterion(outputs, y_batch)  # y_batch: [B]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)  # 保存每个 epoch 的损失

        # 使用原始验证集（不做平衡采样）
        acc, auc = evaluate_model(
            model,
            val_loader,
            device,
            output_dir=None,
            is_best=False,
            verbose=True,
            balance=False
        )
        val_accuracies.append(acc)
        val_aucs.append(auc)

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}  "
              f"Val Acc: {acc:.4f}  Val AUC: {auc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1  # 记录最佳 epoch 的索引
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_model.pth"))
            print(f"模型已保存：best_model.pth at epoch {best_epoch}")

            # 保存最优模型时也使用原始验证集评估
            _ = evaluate_model(
                model,
                val_loader,
                device,
                output_dir=output_dir,
                is_best=True,
                verbose=False,
                balance=False
            )

    # 输出训练结束时最好的 epoch 的索引
    print(f"最好的模型是第 {best_epoch} 个 epoch，准确度为 {best_acc:.4f}")
    return train_losses, val_accuracies, val_aucs, best_epoch
