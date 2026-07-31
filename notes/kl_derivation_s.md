# KL 散度推导：从定义到 VAE 的解析解

## 起点：KL 散度的定义

KL 散度衡量两个分布 $q(z)$ 和 $p(z)$ 之间的"距离"：

$$D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz = \mathbb{E}_{z \sim q}\left[\log q(z) - \log p(z)\right]$$

在 VAE 里：
- $q(z|x) = \mathcal{N}(z; \mu, \sigma^2)$ —— 编码器输出的分布（后验）
- $p(z) = \mathcal{N}(z; 0, I)$ —— 先验（标准正态）

目标是：**让 $q(z|x)$ 靠近 $p(z)$**。

---

## 第一步：写出两个高斯分布的概率密度

$D$ 维高斯分布（对角协方差）的概率密度函数：

$$\mathcal{N}(z; \mu, \Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(z - \mu)^T \Sigma^{-1} (z - \mu)\right)$$

取对数：

$$\log \mathcal{N}(z; \mu, \Sigma) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(z - \mu)^T \Sigma^{-1} (z - \mu)$$

**对于 $q(z|x)$**（对角协方差，每维方差 $\sigma_j^2$）：

$$\log q(z|x) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_{j=1}^{D}\log\sigma_j^2 - \frac{1}{2}\sum_{j=1}^{D}\frac{(z_j - \mu_j)^2}{\sigma_j^2}$$

**对于 $p(z) = \mathcal{N}(0, I)$**（$\mu=0$, $\sigma^2=1$）：

$$\log p(z) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_{j=1}^{D} z_j^2$$

---

## 第二步：计算对数比 $\log q(z|x) - \log p(z)$

两式相减，$-\frac{D}{2}\log(2\pi)$ 被消掉：

$$\log q(z|x) - \log p(z) = -\frac{1}{2}\sum_{j=1}^{D}\log\sigma_j^2 - \frac{1}{2}\sum_{j=1}^{D}\frac{(z_j - \mu_j)^2}{\sigma_j^2} + \frac{1}{2}\sum_{j=1}^{D} z_j^2$$

---

## 第三步：对 $z \sim q$ 取期望

$$D_{\text{KL}}(q \| p) = \mathbb{E}_{z \sim q}\left[\log q(z|x) - \log p(z)\right]$$

利用期望的线性性，逐项计算。**先看单维**（下标 $j$），最后再求和：

$$\text{第 } j \text{ 维的贡献} = -\frac{1}{2}\log\sigma_j^2 - \frac{1}{2}\mathbb{E}_q\left[\frac{(z_j - \mu_j)^2}{\sigma_j^2}\right] + \frac{1}{2}\mathbb{E}_q[z_j^2]$$

分别计算三个期望：

**期望 ①**：$\mathbb{E}_q\left[\frac{(z_j - \mu_j)^2}{\sigma_j^2}\right]$

由于 $z_j \sim \mathcal{N}(\mu_j, \sigma_j^2)$，所以 $\mathbb{E}_q[(z_j - \mu_j)^2] = \sigma_j^2$（方差的定义）。

$$\mathbb{E}_q\left[\frac{(z_j - \mu_j)^2}{\sigma_j^2}\right] = \frac{\sigma_j^2}{\sigma_j^2} = 1$$

**期望 ②**：$\mathbb{E}_q[z_j^2]$

利用方差公式 $\mathbb{E}[z_j^2] = \text{Var}(z_j) + (\mathbb{E}[z_j])^2 = \sigma_j^2 + \mu_j^2$：

$$\mathbb{E}_q[z_j^2] = \sigma_j^2 + \mu_j^2$$

---

## 第四步：代入，化单维结果

$$\text{第 } j \text{ 维} = -\frac{1}{2}\log\sigma_j^2 - \frac{1}{2}(1) + \frac{1}{2}(\sigma_j^2 + \mu_j^2)$$

$$= -\frac{1}{2}\log\sigma_j^2 - \frac{1}{2} + \frac{1}{2}\sigma_j^2 + \frac{1}{2}\mu_j^2$$

$$= \frac{1}{2}\left(-1 - \log\sigma_j^2 + \mu_j^2 + \sigma_j^2\right)$$

$$= -\frac{1}{2}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

## 第五步：求和，得到最终公式

对 $D$ 维求和：

$$D_{\text{KL}}\left(q_\phi(z | x) \| p(z)\right) = -\frac{1}{2} \sum_{j=1}^{D} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

## 回答你的问题

### "-1/2 是？"

不是。$-\frac{1}{2}$ 是**代数化简的自然结果**，不是人为加的正则系数。它来自高斯分布 PDF 里的 $-\frac{1}{2}$ 系数，推导过程中一直带着。

### "是代到正态分布式子里吗？"

对，完整路径是：

> KL 定义 → 代入 $q$ 和 $p$ 的高斯 PDF → 取对数 → 取期望 → 利用高斯矩公式化简 → 得到解析解

关键的一步是**期望的计算**：因为是对 $q$ 取期望，而 $q$ 本身就是高斯，所以 $\mathbb{E}[(z-\mu)^2] = \sigma^2$ 和 $\mathbb{E}[z^2] = \sigma^2 + \mu^2$ 可以直接算出来，不需要采样近似。这就是为什么叫"解析解" —— 有闭式表达式，不需要蒙特卡洛估计。

### "为什么是这个式子？"

直觉理解最终公式里每一项的作用：

$$-\frac{1}{2}\left(\underbrace{1}_{\text{常数}} + \underbrace{\log\sigma_j^2}_{\text{方差小→负得多}} - \underbrace{\mu_j^2}_{\text{均值偏离→惩罚}} - \underbrace{\sigma_j^2}_{\text{方差大→惩罚}}\right)$$

- $\mu_j = 0$, $\sigma_j^2 = 1$ 时：$1 + 0 - 0 - 1 = 0$，KL = 0（完美匹配先验）
- $\mu_j$ 偏离 0 → $-\mu_j^2$ 让 KL 增大 → **惩罚均值偏离原点**
- $\sigma_j^2$ 偏离 1 → 无论太大还是太小，KL 都增大 → **惩罚方差偏离 1**

所以 KL 项的作用：**把每个样本的隐变量分布拉向标准正态 $\mathcal{N}(0, I)$**，让隐空间连续、可插值、可采样。
