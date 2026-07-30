# VAE — Variational Autoencoder on MNIST

基于 **VAE (Variational Autoencoder)** 的 MNIST 手写数字生成模型，包含完整的训练与推理流程。

编码器-解码器采用**全连接 (MLP)** 架构，损失函数为标准 VAE 的 ELBO 目标 (重建损失 + KL 散度)。

---

## Copyright

```
Copyright (c) 2026 HolmHuang
```

---

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 训练

```bash
python train_mnist.py
```

首次运行会自动下载 MNIST 数据集到 `./data`。训练完成后，模型权重保存在 `./outputs/weights/model_final.pth`。

常用参数：

| 参数              | 默认值    | 说明                             |
| ----------------- | --------- | -------------------------------- |
| `--epochs`        | 100       | 训练轮数                         |
| `--batch_size`    | 128       | 批大小                           |
| `--lr`            | 1e-3      | 学习率                           |
| `--latent_dim`    | 20        | 隐变量维度                       |
| `--num_img`       | 0         | 训练图像数量 (0 = 全部 60000 张) |
| `--save_dir`      | ./outputs | 输出根目录                       |

### 3. 生成样本

```bash
python sample_mnist.py
```

生成结果保存在 `./outputs/samples/`，包含单张样本和拼接好的网格大图。

常用参数：

| 参数            | 默认值                            | 说明             |
| --------------- | --------------------------------- | ---------------- |
| `--checkpoint`  | ./outputs/weights/model_final.pth | 模型权重路径     |
| `--n_samples`   | 16                                | 生成样本数量     |
| `--reconstruct` | False                             | 测试集重建可视化 |
| `--interpolate` | False                             | 隐空间插值       |
| `--manifold`    | False                             | 隐空间流形可视化 |
| `--all`         | False                             | 执行全部功能     |

---

## 模型架构

### MLP-VAE 编码器

```
Input (784)
  └─► Linear(784→512) → ReLU
        └─► Linear(512→256) → ReLU
              ├─► fc_mu     → μ  (20-dim)
              └─► fc_logvar → log σ² (20-dim)
```

### MLP-VAE 解码器

```
z (20-dim)
  └─► Linear(20→256) → ReLU
        └─► Linear(256→512) → ReLU
              └─► Linear(512→784) → Sigmoid
                    Output (784)
```

---

## 公式详解

本项目基于 VAE 原始论文 (Kingma & Welling, 2014) 实现。以下公式与代码变量一一对应，方便对照查看。

### 符号说明

| 符号            | 代码变量             | 含义                                  |
| --------------- | -------------------- | ------------------------------------- |
| $x$             | `images`             | 输入图像                              |
| $\hat{x}$       | `recon`              | 重建图像                              |
| $\mu$           | `mu`                 | 编码器输出的均值                      |
| $\log \sigma^2$ | `logvar`             | 编码器输出的对数方差                  |
| $z$             | `z` (reparameterize) | 采样的隐变量                          |
| $\epsilon$      | `eps`                | 采样自 $\mathcal{N}(0, I)$ 的高斯噪声 |
| $D$             | `latent_dim`         | 隐变量维度                            |

---

### 一、训练

#### 1. 编码器 $q_\phi(z | x)$

编码器输出隐变量分布的参数 $\mu$ 和 $\log \sigma^2$：

```python
h = F.relu(self.fc1(x))      # 784 → 512
h = F.relu(self.fc2(h))      # 512 → 256
mu     = self.fc_mu(h)       # μ
logvar = self.fc_logvar(h)   # log σ²
```

$$q_\phi(z | x) = \mathcal{N}\left(z; \; \mu_\phi(x), \; \text{diag}(\sigma^2_\phi(x))\right)$$

#### 2. 重参数化技巧 (Reparameterization Trick)

为了使采样操作可微，将随机性转移到外部噪声变量 $\epsilon$：

```python
std = torch.exp(0.5 * logvar)    # σ = exp(0.5 · log σ²)
eps = torch.randn_like(std)      # ε ~ N(0, I)
z = mu + eps * std               # z = μ + ε·σ
```

$$z = \mu + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$$

#### 3. 解码器 $p_\theta(x | z)$

从隐变量 $z$ 重建原始图像：

```python
h = F.relu(self.fc3(z))          # 20 → 256
h = F.relu(self.fc4(h))          # 256 → 512
recon = torch.sigmoid(self.fc5(h))  # 512 → 784, 输出 [0, 1]
```

$$p_\theta(x | z) = \text{Bernoulli}\left(x; \; \hat{x}_\theta(z)\right)$$

#### 4. 损失函数

VAE 的优化目标为最大化 ELBO (Evidence Lower Bound)，等价于最小化：

```python
recon_loss = F.mse_loss(recon, original, reduction='sum') / batch_size
kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
total = recon_loss + kl_div
```

$$\mathcal{L}(\theta, \phi; x) = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{重建损失}} + \underbrace{D_{\text{KL}}\left(q_\phi(z | x) \| p(z)\right)}_{\text{KL 散度}}$$

**重建损失** (MSE 版本)：

$$\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \left\| x_i - \hat{x}_i \right\|^2$$

**KL 散度** (高斯分布有解析解)：

$$D_{\text{KL}}\left(q_\phi(z | x) \| p(z)\right) = -\frac{1}{2} \sum_{j=1}^{D} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

> 代码与公式的逐项对应见 [notes/kl_divergence.md](notes/kl_divergence.md)，完整推导见 [notes/kl_derivation.md](notes/kl_derivation.md)。

---

### 二、采样（推理）

#### 1. 随机生成

从先验分布 $p(z) = \mathcal{N}(0, I)$ 采样隐变量，通过解码器生成新图像：

```python
z = torch.randn(n, latent_dim)   # z ~ N(0, I)
generated = model.decode(z)      # p(x|z)
```

$$z \sim p(z) = \mathcal{N}(0, I), \quad \hat{x} = p_\theta(x | z)$$

#### 2. 隐空间插值

对两个输入 $x_1, x_2$ 分别编码得到 $\mu_1, \mu_2$，在隐空间线性插值后解码：

```python
z_interp = (1 - α) * μ1 + α * μ2
x_interp = model.decode(z_interp)
```

$$z_\alpha = (1 - \alpha) \cdot \mu_1 + \alpha \cdot \mu_2, \quad \alpha \in [0, 1]$$

#### 3. 隐空间流形

当 $D \geq 2$ 时，固定其他维度为 0，在 $z[0]$ 和 $z[1]$ 方向均匀采样并解码，可视化隐空间的二维流形。

---

### 三、ELBO 推导

VAE 的目标是最大化数据的对数似然 $\log p_\theta(x)$。引入变分分布 $q_\phi(z | x)$ 后：

$$\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] + D_{\text{KL}}\left(q_\phi(z|x) \| p_\theta(z|x)\right)$$

由于 KL 散度非负，得到下界 (ELBO)：

$$\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right]}_{\text{重建项}} - \underbrace{D_{\text{KL}}\left(q_\phi(z|x) \| p(z)\right)}_{\text{先验匹配项}}$$

最大化 ELBO 等价于最小化代码中的损失函数：

$$\mathcal{L} = -\text{ELBO} = \mathcal{L}_{\text{recon}} + D_{\text{KL}}$$

---

### 四、符号与代码对照表

| 数学符号                     | 代码变量                           | 含义                         |
| ---------------------------- | ---------------------------------- | ---------------------------- |
| $q_\phi(z \| x)$             | `model.encode(x)`                  | 变分后验 (编码器)            |
| $\mu_\phi(x)$                | `mu`                               | 后验均值                     |
| $\log \sigma^2_\phi(x)$      | `logvar`                           | 后验对数方差                 |
| $z = \mu + \epsilon \sigma$  | `reparameterize(mu, logvar)`       | 重参数化采样                 |
| $p_\theta(x \| z)$           | `model.decode(z)`                  | 似然 (解码器)                |
| $p(z)$                       | `torch.randn(..., latent_dim)`     | 先验分布 $\mathcal{N}(0, I)$ |
| $\mathcal{L}_{\text{recon}}$ | `F.mse_loss(recon, original)`      | 重建损失                     |
| $D_{\text{KL}}$              | `-0.5 * sum(1 + logvar - μ² - σ²)` | KL 散度                      |

---

## 项目结构

```
VAE-MNIST/
├── train_mnist.py              # 训练脚本
├── sample_mnist.py             # 推理/采样脚本
├── pyproject.toml              # 项目配置 (uv)
├── data/                       # MNIST 数据集 (自动下载)
├── outputs/                    # 训练输出
│   ├── weights/                # 模型权重
│   │   ├── best_model.pth      # 最优模型
│   │   └── model_final.pth     # 最终模型
│   ├── losses/                 # 损失曲线数据 (.npy)
│   └── img/                    # 训练过程可视化
│       ├── reconstructions/    # 重建对比图
│       └── samples/            # 随机采样图
├── notes/                      # 学习笔记
│   ├── kl_divergence.md        # KL 散度：代码与公式对应
│   └── kl_derivation.md        # KL 散度：从定义到解析解的完整推导
└── README.md
```

---

## 参考

- 论文: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2014)
