import os
import torch
from torch.utils.data import DataLoader
from model2 import CNNTransformerClassifier
from dataset import IMUDataset
from config import input_dim, seq_len, num_classes, d_model, nhead, num_layers, dropout, device
from evaluate import evaluate_model
from scipy.io import savemat

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np

def plot_test_visualizations(labels, preds, probs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # ROC 曲线 & AUC
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # Precision–Recall 曲线 & PR AUC
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.title("Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=600)
    plt.close()

    # 预测概率分布直方图
    plt.figure(figsize=(6, 5))
    plt.hist(probs, bins=20, range=(0, 1), alpha=0.7, color='steelblue', edgecolor='black')
    plt.title("Predicted Probability Distribution")
    plt.xlabel("Probability for Class 1")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prob_histogram.png"), dpi=600)
    plt.close()

def evaluate_test(model_path, test_mat_path, output_dir, test_set_name):
    print(f"[INFO] Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model = CNNTransformerClassifier(input_dim, seq_len, num_classes, d_model, nhead, num_layers, dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 加载测试集（开启 extra 字段加载）
    test_dataset = IMUDataset(test_mat_path, load_extra=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型评估
    test_acc, test_auc = evaluate_model(
        model, test_loader, device,
        output_dir=output_dir,
        is_best=False,
        verbose=True,
        tag='test'
    )

    # 收集预测结果
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)  # [B]
            probs = torch.sigmoid(outputs).cpu().numpy()  # sigmoid概率
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    # 保存为 .mat 文件
    result_mat = {
        "test_accuracy":     test_acc,
        "test_auc":          test_auc,
        "true_labels":       np.array(all_labels),
        "predicted_labels":  np.array(all_preds),
        "predicted_probs":   np.array(all_probs)
    }

    for k, v in test_dataset.extra.items():
        result_mat[k] = v

    save_path = os.path.join(output_dir, f"test_{test_set_name}.mat")
    savemat(save_path, result_mat)
    print(f"[INFO] 测试结果已保存至: {save_path}")

    # 可视化
    viz_dir = os.path.join(output_dir, "visualizations")
    plot_test_visualizations(all_labels, all_preds, all_probs, viz_dir)
    print(f"[INFO] 测试可视化已保存至: {viz_dir}")

if __name__ == "__main__":
    output_path = r'C:\Users\swl\Desktop\实验数据\pythonCode\onlyIMU\output'
    train_result_dir = 'run_36'
    test_sl = 200
    mission = 'turn2'
    base_dir = os.path.join(output_path, train_result_dir)

    test_set_name = f'DWBH_turn_4_tensor_{test_sl}'
    # turn
    # DWBH_turn_4_tensor_
    # JJGC_turn_2_tensor_
    # LLP_turn_10_tensor_
    # YLW_turn_2_tensor_
    # test_set_name = f'combined_tensor_data'
    # combined_tensor_data

    # slop
    # DWBH_slop_4_tensor_
    # JJGC_slop_4_tensor_
    # LLP_slop_10_tensor_
    # YLW_slop_2_tensor_


    test_mat_file = os.path.join(
        fr"C:\Users\swl\Desktop\实验数据\onlyIMU\tensor\{mission}_tensor\模式1\{test_sl}\test",
        test_set_name + ".mat"
    )
    test_output_dir = os.path.join(base_dir, 'test_result_' + test_set_name)

    evaluate_test(
        model_path=os.path.join(output_path, train_result_dir, "best_model.pth"),
        test_mat_path=test_mat_file,
        output_dir=test_output_dir,
        test_set_name=test_set_name
    )
