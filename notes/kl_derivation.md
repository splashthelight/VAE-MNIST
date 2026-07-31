# KL 散度推导：从单变量到多变量

## 起点：KL 散度的定义

KL 散度衡量两个分布 $q(z)$ 和 $p(z)$ 之间的"距离"：

$$D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz = \mathbb{E}_{z \sim q}\left[\log q(z) - \log p(z)\right]$$

在 VAE 里：
- $q(z|x) = \mathcal{N}(z; \mu, \sigma^2)$ —— 编码器输出的分布（后验）
- $p(z) = \mathcal{N}(z; 0, I)$ —— 先验（标准正态）

目标是：**让 $q(z|x)$ 靠近 $p(z)$**。

---

## 第一步：单变量推导

先只看**一维**情况：$q = \mathcal{N}(\mu, \sigma^2)$，$p = \mathcal{N}(0, 1)$。

两个正态分布的 PDF：

$$q(z) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)$$

$$p(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)$$

代入 KL 定义：

$$D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz$$

先算对数比：

$$\log \frac{q(z)}{p(z)} = \log q(z) - \log p(z)$$

$$= \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}\right] - \left[-\frac{1}{2}\log(2\pi) - \frac{z^2}{2}\right]$$

$$= -\frac{1}{2}\log\sigma^2 - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}$$

现在对 $z \sim q$ 取期望：

$$D_{\text{KL}}(q \| p) = \mathbb{E}_q\left[-\frac{1}{2}\log\sigma^2 - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}\right]$$

利用期望的线性性，逐项计算：

**第 1 项**：$\mathbb{E}_q\left[-\frac{1}{2}\log\sigma^2\right] = -\frac{1}{2}\log\sigma^2$（常数）

**第 2 项**：$\mathbb{E}_q\left[-\frac{(z-\mu)^2}{2\sigma^2}\right] = -\frac{1}{2\sigma^2}\mathbb{E}_q[(z-\mu)^2] = -\frac{1}{2\sigma^2} \cdot \sigma^2 = -\frac{1}{2}$

（这里用了方差定义：$\mathbb{E}[(z-\mu)^2] = \sigma^2$）

**第 3 项**：$\mathbb{E}_q\left[\frac{z^2}{2}\right] = \frac{1}{2}\mathbb{E}_q[z^2] = \frac{1}{2}(\sigma^2 + \mu^2)$

（这里用了 $\mathbb{E}[z^2] = \text{Var}(z) + (\mathbb{E}[z])^2 = \sigma^2 + \mu^2$）

三项相加：

$$D_{\text{KL}}(q \| p) = -\frac{1}{2}\log\sigma^2 - \frac{1}{2} + \frac{1}{2}(\sigma^2 + \mu^2)$$

$$= \frac{1}{2}\left(-1 - \log\sigma^2 + \mu^2 + \sigma^2\right)$$

$$= -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

---

## 第二步：推广到 D 维

VAE 的隐变量是 $D$ 维向量 $z = (z_1, z_2, \ldots, z_D)$。

关键：**对角协方差 = 各维度独立**，所以 $D$ 维联合 PDF 可分解为 $D$ 个独立单变量 PDF 的乘积：

$$q(z|x) = \prod_{j=1}^{D} q_j(z_j), \quad q_j = \mathcal{N}(\mu_j, \sigma_j^2)$$

$$p(z) = \prod_{j=1}^{D} p_j(z_j), \quad p_j = \mathcal{N}(0, 1)$$

代入 KL 定义：

$$D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz = \int q(z) \left[\sum_{j=1}^{D} \log \frac{q_j(z_j)}{p_j(z_j)}\right] dz$$

由期望的线性性，求和与积分可交换：

$$= \sum_{j=1}^{D} \int q_j(z_j) \log \frac{q_j(z_j)}{p_j(z_j)} \, dz_j = \sum_{j=1}^{D} D_{\text{KL}}(q_j \| p_j)$$

**D 维 KL = D 个单变量 KL 之和**。

把第一步的单变量结果代入：

$$D_{\text{KL}}\left(q_\phi(z | x) \| p(z)\right) = -\frac{1}{2} \sum_{j=1}^{D} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

## 与代码的对应

```python
kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
```

| 公式项 | 代码 | 含义 |
|---|---|---|
| $-\frac{1}{2}$ | `-0.5 *` | 前置系数 |
| $\sum_{j=1}^{D}$ | `torch.sum(..., dim=1)` | 沿隐变量维度求和 |
| $1$ | `1` | 常数 |
| $\log \sigma_j^2$ | `logvar` | 对数方差 |
| $\mu_j^2$ | `mu.pow(2)` | 均值平方 |
| $\sigma_j^2$ | `logvar.exp()` | 方差 |

---

## 直觉

最终公式里每一项的作用：

$$-\frac{1}{2}\left(\underbrace{1}_{\text{常数}} + \underbrace{\log\sigma_j^2}_{\text{方差小→负得多}} - \underbrace{\mu_j^2}_{\text{均值偏离→惩罚}} - \underbrace{\sigma_j^2}_{\text{方差大→惩罚}}\right)$$

- $\mu_j = 0$, $\sigma_j^2 = 1$ 时：$1 + 0 - 0 - 1 = 0$，KL = 0（完美匹配先验）
- $\mu_j$ 偏离 0 → $-\mu_j^2$ 让 KL 增大 → **惩罚均值偏离原点**
- $\sigma_j^2$ 偏离 1 → 无论太大还是太小，KL 都增大 → **惩罚方差偏离 1**

所以 KL 项的作用：**把每个样本的隐变量分布拉向标准正态 $\mathcal{N}(0, I)$**，让隐空间连续、可插值、可采样。
