import matplotlib.pyplot as plt
import math

# 指定支持中文和全角符号的字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 保证负号 '-' 正常显示

import numpy as np
import torch
import os
from sklearn.manifold import TSNE


def plot_training_curves(losses, accs, aucs, output_dir):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Loss Curve")

    plt.subplot(1, 3, 2)
    plt.plot(accs)
    plt.title("Validation Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(aucs)
    plt.title("Validation AUC")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=350)
    plt.close()


def plot_attention_weights(model, sample_input, device, output_dir, layers=[1, 4, 8, 'last']):
    """
    同时可视化 Transformer 指定层的注意力权重。默认绘制第1层、第4层、第8层和最后一层。
    如果某层编号超过实际层数，将被跳过。
    """
    model.eval()
    with torch.no_grad():
        x = sample_input.unsqueeze(0).to(device)  # [1, T, features]
        feat = model.feature_extractor(x)         # [1, d_model, T]
        feat = feat.permute(0, 2, 1)              # [1, T, d_model]
        cls = model.cls_token.expand(1, -1, -1).to(device)
        tokens = torch.cat([cls, feat], dim=1)    # [1, T+1, d_model]
        tokens = model.pos_encoder(tokens)

        total_layers = len(model.transformer_encoder.layers)
        to_plot = []
        for lvl in layers:
            if lvl == 'last':
                layer = model.transformer_encoder.layers[-1]
                title = 'Layer Last'
                to_plot.append((layer, title))
            elif isinstance(lvl, int) and 1 <= lvl <= total_layers:
                layer = model.transformer_encoder.layers[lvl - 1]
                title = f'Layer {lvl}'
                to_plot.append((layer, title))
            else:
                print(f"Warning: layer {lvl} is out of range (1-{total_layers}), skipping.")

        attn_maps = []
        for layer, title in to_plot:
            _, attn_w = layer.self_attn(tokens, tokens, tokens)
            attn_maps.append((attn_w[0].cpu().numpy(), title))

    # 绘制多个子图
    n = len(attn_maps)
    cols = 2
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 5, rows * 4))
    for i, (attn, title) in enumerate(attn_maps):
        ax = plt.subplot(rows, cols, i + 1)
        im = ax.imshow(attn, aspect='auto', origin='lower')
        ax.set_title(f"Attention ({title})")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Sequence Position")
        plt.colorbar(im, ax=ax, label='Attention')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "attention_evolution.png"), dpi=350, bbox_inches='tight')
    plt.close()


def plot_tsne_with_misclass(model, data_loader, device, output_dir):
    model.eval()
    all_feats, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            B = X_batch.size(0)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            feat = model.feature_extractor(X_batch)  # [B, d_model, T]
            feat = feat.permute(0, 2, 1)             # [B, T, d_model]
            cls = model.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, feat], dim=1)
            tokens = model.pos_encoder(tokens)
            enc_out = model.transformer_encoder(tokens)

            cls_out = enc_out[:, 0, :]  # [B, d_model]
            logits = model.fc(model.norm(cls_out)).squeeze(1)  # [B]
            probs = torch.sigmoid(logits)                      # [B]
            preds = (probs > 0.5).long()

            all_feats.append(cls_out.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(feats)

    correct = labels == preds
    wrong = ~correct

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[correct, 0], reduced[correct, 1],
                c='red', marker='o', alpha=0.7, label='Correct')
    plt.scatter(reduced[wrong, 0], reduced[wrong, 1],
                c='black', marker='x', label='Wrong')

    plt.title("t-SNE Feature Distribution (o: Correct, X: Incorrect)")
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tsne_misclassified.png"), dpi=350, bbox_inches='tight')
    plt.close()
