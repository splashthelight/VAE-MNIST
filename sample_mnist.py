# -*- coding: utf-8 -*-
"""
MNIST 变分自编码器 (VAE) 推理/采样脚本
========================================

加载训练好的 MNIST VAE 模型，实现:
  1. 随机采样生成新数字
  2. 测试集重建可视化
  3. 隐空间插值 (interpolation)
  4. 2D 隐空间流形可视化 (latent manifold)

使用方法:
    python sample_mnist.py
    python sample_mnist.py --checkpoint ./outputs/weights/model_final.pth --n_samples 16
    python sample_mnist.py --interpolate            # 隐空间插值
    python sample_mnist.py --manifold               # 隐空间流形 (需 latent_dim >= 2)
    python sample_mnist.py --all                    # 执行全部功能
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 直接引用训练脚本中的模型定义
from train_mnist import VAE


# ============================================================================
# 辅助函数
# ============================================================================

def _to_vec(images):
    """将图像张量展平为向量: (B,1,28,28) → (B,784)"""
    return images.view(images.size(0), -1)


def _to_img(vectors):
    """将向量重塑为图像: (B,784) → (B,1,28,28)"""
    return vectors.view(-1, 1, 28, 28)


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Sample / generate from MNIST VAE")
    parser.add_argument("--checkpoint", type=str, default="./outputs/weights/model_final.pth",
                        help="模型权重路径")
    parser.add_argument("--latent_dim", type=int, default=20, help="隐变量维度 (需与训练时一致)")
    parser.add_argument("--n_samples", type=int, default=16, help="随机采样数量")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--data_root", type=str, default="./data", help="MNIST 数据目录")
    parser.add_argument("--save_dir", type=str, default="./outputs/samples", help="样本保存目录")
    parser.add_argument("--interpolate", action="store_true", help="执行隐空间插值")
    parser.add_argument("--manifold", action="store_true", help="执行隐空间流形可视化")
    parser.add_argument("--reconstruct", action="store_true", help="执行测试集重建")
    parser.add_argument("--all", action="store_true", help="执行全部功能")
    return parser.parse_args()


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.all:
        args.interpolate = args.manifold = args.reconstruct = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 加载模型 --------------------
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {checkpoint_path}")

    # 尝试从 checkpoint 中读取 latent_dim
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    latent_dim = ckpt.get('latent_dim', args.latent_dim)

    model = VAE(input_dim=784, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Info] 已加载模型: {checkpoint_path}  (latent_dim={latent_dim})")

    # -------------------- 1. 随机采样 --------------------
    print("\n[1/4] 随机采样生成...")
    all_samples = []
    n_batches = int(np.ceil(args.n_samples / args.batch_size))
    with torch.no_grad():
        for _ in range(n_batches):
            n = min(args.batch_size, args.n_samples - len(all_samples))
            z = torch.randn(n, latent_dim, device=device)
            generated = model.decode(z)
            all_samples.append(generated.cpu())
    samples = torch.cat(all_samples, dim=0)[:args.n_samples]

    # 单独保存 + 网格大图 (reshape 为图像格式)
    samples_img = _to_img(samples)
    single_dir = save_dir / "single"
    single_dir.mkdir(exist_ok=True)
    for i, img in enumerate(samples_img):
        _save_single(img[0], single_dir / f"sample_{i:04d}.png")
    _save_grid(samples_img, save_dir / "grid_random.png")
    print(f"  → 已生成 {len(samples)} 张样本，保存到 {save_dir}")

    # -------------------- 2. 测试集重建 --------------------
    if args.reconstruct:
        print("\n[2/4] 测试集重建...")
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        all_recons = []
        all_origs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                recon, _, _ = model(_to_vec(images))
                all_recons.append(recon.cpu())
                all_origs.append(images.cpu())
        recons = torch.cat(all_recons, dim=0)
        origs  = torch.cat(all_origs, dim=0)

        # 保存对比图 (重建输出为 784 向量，需 reshape 为图像)
        _save_recon_comparison(origs[:64], _to_img(recons[:64]), save_dir / "reconstructions.png")
        print(f"  → 重建对比图已保存到 {save_dir / 'reconstructions.png'}")

    # -------------------- 3. 隐空间插值 --------------------
    if args.interpolate:
        print("\n[3/4] 隐空间插值...")
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        with torch.no_grad():
            # 取两张不同的测试图
            img1, _ = next(iter(test_loader))
            img2, _ = next(iter(test_loader))
            img1 = img1.to(device)
            img2 = img2.to(device)

            mu1, _ = model.encode(_to_vec(img1))
            mu2, _ = model.encode(_to_vec(img2))

            # 线性插值
            n_steps = 10
            alphas = torch.linspace(0, 1, n_steps, device=device)
            interpolations = []
            for a in alphas:
                z = (1 - a) * mu1 + a * mu2
                recon = model.decode(z)
                interpolations.append(recon.cpu())

        _save_interpolation(img1.cpu(), img2.cpu(), interpolations, save_dir / "interpolation.png")
        print(f"  → 插值图已保存到 {save_dir / 'interpolation.png'}")

    # -------------------- 4. 隐空间流形 --------------------
    if args.manifold:
        if latent_dim < 2:
            print(f"\n[4/4] 跳过隐空间流形 (latent_dim={latent_dim} < 2)")
        else:
            print("\n[4/4] 隐空间流形可视化...")
            _visualize_manifold(model, latent_dim, device, save_dir)
            print(f"  → 流形图已保存到 {save_dir / 'latent_manifold.png'}")

    print("\n[Done] 全部完成!")


# ============================================================================
# 可视化辅助函数
# ============================================================================

def _save_single(tensor, path):
    """保存单张灰度图"""
    plt.figure(figsize=(2, 2))
    plt.imshow(tensor.numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def _save_grid(samples, path, n=None):
    """拼接样本为网格大图"""
    n = n or len(samples)
    n = min(n, len(samples))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            ax.imshow(samples[i, 0].numpy(), cmap='gray')
        ax.axis('off')

    fig.tight_layout(pad=0.2)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _save_recon_comparison(originals, reconstructions, path, n=64):
    """保存原图与重建图对比"""
    n_show = min(n, len(originals))
    cols = 8
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(2 * rows, cols, figsize=(cols * 1.0, 2 * rows * 1.0))
    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        if idx < n_show:
            axes[2 * r, c].imshow(originals[idx, 0], cmap='gray')
            axes[2 * r + 1, c].imshow(reconstructions[idx, 0], cmap='gray')
        axes[2 * r, c].axis('off')
        axes[2 * r + 1, c].axis('off')
    # 标注
    axes[0, 0].set_title('Original', fontsize=8, loc='left')
    axes[1, 0].set_title('Reconstructed', fontsize=8, loc='left')

    fig.tight_layout(pad=0.2)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _save_interpolation(img1, img2, interpolations, path):
    """保存隐空间插值结果 (插值输出为 784 向量，需 reshape)"""
    n_steps = len(interpolations)
    fig, axes = plt.subplots(1, n_steps + 2, figsize=( (n_steps + 2) * 1.2, 1.5))

    axes[0].imshow(img1[0, 0], cmap='gray')
    axes[0].set_title('Start', fontsize=8)
    axes[0].axis('off')

    for i, recon in enumerate(interpolations):
        axes[i + 1].imshow(recon[0].reshape(28, 28), cmap='gray')
        axes[i + 1].axis('off')

    axes[-1].imshow(img2[0, 0], cmap='gray')
    axes[-1].set_title('End', fontsize=8)
    axes[-1].axis('off')

    fig.suptitle('Latent Space Interpolation', fontsize=12)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _visualize_manifold(model, latent_dim, device, save_dir, n=15, scale=2.0):
    """可视化隐空间前两维的解码流形"""
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    model.eval()
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.zeros(1, latent_dim, device=device)
                z[0, 0] = xi
                z[0, 1] = yi
                x_decoded = model.decode(z)
                digit = x_decoded.cpu().reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.title('Latent Space Manifold (first 2 dims)', fontsize=14)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.tight_layout()
    plt.savefig(str(save_dir / "latent_manifold.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
