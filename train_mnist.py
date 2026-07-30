# -*- coding: utf-8 -*-
"""
MNIST 变分自编码器 (VAE) 训练脚本
=================================

基于 Variational Autoencoder (Kingma & Welling, 2014)
在 MNIST 手写数字数据集上训练 MLP-VAE 模型。

编码器-解码器采用全连接 (MLP) 架构，损失函数为标准 VAE 的
ELBO 目标: 重建损失 (MSE) + KL 散度。

使用方法:
    python train_mnist.py
    python train_mnist.py --epochs 100 --batch_size 128 --lr 1e-3 --latent_dim 20
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# 模型定义: MLP-VAE
# ============================================================================

class VAE(nn.Module):
    """多层感知机变分自编码器

    编码器: 784 → 512 → 256 → μ / logσ²
    解码器: 20  → 256 → 512 → 784
    """

    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # ----- 编码器 -----
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu    = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # ----- 解码器 -----
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧: z = μ + ε·σ,  ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ============================================================================
# 损失函数
# ============================================================================

def vae_loss(recon, original, mu, logvar):
    """VAE 总损失 = 重建损失 + KL 散度

    - 重建损失: MSE，对像素求和后除以 batch 大小
    - KL 散度:  D_KL(q(z|x) || p(z)) = -0.5 * Σ(1 + logσ² - μ² - σ²)
    """
    # 重建损失 (MSE, sum over pixels, 除以 batch 大小取平均)
    recon_loss = F.mse_loss(recon, original, reduction='sum') / original.size(0)

    # KL 散度 (对每个样本求和，再取 batch 平均)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = kl_div.mean()

    total = recon_loss + kl_div
    return total, recon_loss, kl_div


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP-VAE on MNIST")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--latent_dim", type=int, default=20, help="隐变量维度")
    parser.add_argument("--num_img", type=int, default=0, help="使用的训练图像数量 (0 = 全部 60000 张)")
    parser.add_argument("--vis_interval", type=int, default=5, help="可视化间隔 (每 N epoch)")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔 (每 N epoch)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--data_root", type=str, default="./data", help="MNIST 数据下载目录")
    parser.add_argument("--save_dir", type=str, default="./outputs", help="输出根目录")
    return parser.parse_args()


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    args = parse_args()

    # -------------------- 基础设置 --------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    save_dir = Path(args.save_dir)
    weights_dir  = save_dir / "weights"
    losses_dir   = save_dir / "losses"
    img_recon    = save_dir / "img/reconstructions"
    img_sample   = save_dir / "img/samples"

    for d in [weights_dir, losses_dir, img_recon, img_sample]:
        d.mkdir(parents=True, exist_ok=True)

    # -------------------- 数据加载 --------------------
    transform = transforms.Compose([
        transforms.ToTensor(),  # MNIST 原始 [0, 1]，直接输入
    ])

    train_dataset = datasets.MNIST(
        root=args.data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=args.data_root, train=False, download=True, transform=transform
    )

    if 0 < args.num_img < len(train_dataset):
        indices = torch.arange(args.num_img)
        train_dataset = Subset(train_dataset, indices)
        print(f"[Info] 仅使用前 {args.num_img} 张训练图像")
    else:
        print(f"[Info] 使用全部 {len(train_dataset)} 张训练图像")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # -------------------- 模型与优化器 --------------------
    model = VAE(input_dim=784, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] 模型参数量: {total_params:,}")

    # -------------------- 训练循环 --------------------
    epoch_losses = []     # 每个 epoch 的平均损失
    epoch_recon  = []     # 每个 epoch 的平均重建损失
    epoch_kl     = []     # 每个 epoch 的平均 KL 散度

    best_epoch_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_recon_sum = 0.0
        epoch_kl_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, _ in pbar:
            images = images.view(-1, 784).to(device)  # 展平为向量
            optimizer.zero_grad()

            recon, mu, logvar = model(images)
            loss, recon_loss, kl_div = vae_loss(recon, images, mu, logvar)

            loss.backward()
            optimizer.step()

            epoch_loss_sum  += loss.item()
            epoch_recon_sum += recon_loss.item()
            epoch_kl_sum    += kl_div.item()
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.1f}")

        # 记录 epoch 平均损失
        avg_loss  = epoch_loss_sum / n_batches
        avg_recon = epoch_recon_sum / n_batches
        avg_kl    = epoch_kl_sum / n_batches
        epoch_losses.append(avg_loss)
        epoch_recon.append(avg_recon)
        epoch_kl.append(avg_kl)

        np.save(str(losses_dir / "epoch_losses.npy"), np.array(epoch_losses))
        np.save(str(losses_dir / "epoch_recon.npy"), np.array(epoch_recon))
        np.save(str(losses_dir / "epoch_kl.npy"), np.array(epoch_kl))

        print(f"Epoch [{epoch}/{args.epochs}]  Loss: {avg_loss:.4f}  "
              f"Recon: {avg_recon:.4f}  KL: {avg_kl:.4f}")

        # -------------------- 可视化训练进度 --------------------
        if epoch % args.vis_interval == 0:
            model.eval()
            with torch.no_grad():
                # 重建可视化
                sample_batch, _ = next(iter(test_loader))
                sample_batch = sample_batch.to(device)[:8]
                sample_flat = sample_batch.view(-1, 784)
                recon_batch, _, _ = model(sample_flat)
                recon_batch = recon_batch.view(-1, 1, 28, 28)
                _save_recon_grid(sample_batch, recon_batch, img_recon / f"epoch_{epoch}.png")

                # 随机采样可视化
                z = torch.randn(16, args.latent_dim, device=device)
                generated = model.decode(z).view(-1, 1, 28, 28)
                _save_sample_grid(generated, img_sample / f"epoch_{epoch}.png")

        # -------------------- 保存模型 --------------------
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'latent_dim': args.latent_dim,
                'args': vars(args),
            }, str(weights_dir / f"checkpoint_epoch_{epoch}.pth"))

        # 保存最优模型
        if epoch_losses[-1] < best_epoch_loss:
            best_epoch_loss = epoch_losses[-1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'latent_dim': args.latent_dim,
                'args': vars(args),
            }, str(weights_dir / "best_model.pth"))

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'latent_dim': args.latent_dim,
        'args': vars(args),
    }, str(weights_dir / "model_final.pth"))

    print(f"[Done] 训练完成，最终模型已保存到 {weights_dir / 'model_final.pth'}")
    print(f"[Done] 最优模型已保存到 {weights_dir / 'best_model.pth'} (loss={best_epoch_loss:.4f})")


# ============================================================================
# 可视化辅助函数
# ============================================================================

def _save_recon_grid(originals, reconstructions, path, n=8):
    """保存原图与重建图对比网格"""
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.4))
    for i in range(n):
        axes[0, i].imshow(originals[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i, 0].cpu(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=10, rotation=0, labelpad=50, va='center')
    axes[1, 0].set_ylabel('Recon', fontsize=10, rotation=0, labelpad=50, va='center')
    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_sample_grid(samples, path, n=16):
    """保存随机采样网格"""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            ax.imshow(samples[i, 0].cpu(), cmap='gray')
        ax.axis('off')
    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
