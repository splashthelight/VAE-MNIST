# KL 散度 —— 代码与公式对应

代码：

```python
kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
```

对应公式（两个高斯分布的 KL 散度解析解）：

$$D_{\text{KL}}\left(q_\phi(z | x) \| p(z)\right) = -\frac{1}{2} \sum_{j=1}^{D} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

## 逐项对应

$$-\frac{1}{2} \sum_{j=1}^{D} \left(\underbrace{1}_{\text{常数}} + \underbrace{\log \sigma_j^2}_{\text{logvar}} - \underbrace{\mu_j^2}_{\text{mu.pow(2)}} - \underbrace{\sigma_j^2}_{\text{logvar.exp()}}\right)$$

| 公式项 | 代码 | 含义 |
|---|---|---|
| $-\frac{1}{2}$ | `-0.5 *` | 前置系数 |
| $\sum_{j=1}^{D}$ | `torch.sum(..., dim=1)` | 沿隐变量维度 D 求和 |
| $1$ | `1` | 常数项 |
| $\log \sigma_j^2$ | `logvar` | 编码器输出的对数方差 |
| $\mu_j^2$ | `mu.pow(2)` | 编码器输出的均值平方 |
| $\sigma_j^2$ | `logvar.exp()` | 由 logvar 还原出方差 |

## 直觉

KL 散度衡量 $q(z|x)$（编码器输出的分布）离先验 $p(z) = \mathcal{N}(0, I)$ 有多远：

- 当 $\mu = 0$、$\sigma^2 = 1$ 时，$1 + \log 1 - 0 - 1 = 0$，KL = 0（完全等于先验）
- $\mu$ 偏离 0 → $\mu^2$ 项增大 → KL 增大（惩罚均值偏离原点）
- $\sigma^2$ 偏离 1 → KL 增大（惩罚方差偏离标准正态）

所以 KL 项的作用：**把每个样本的隐变量分布拉向标准正态**，让隐空间连续、可采样。
