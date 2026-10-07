import math
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============ 1. 最小 Transformer 组件 ============
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.d_k, self.h = d_model // h, h
        self.W = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.W(x).view(B, L, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class Sublayer(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, attn, ff, dropout=0.1):
        super().__init__()
        self.attn, self.ff = attn, ff
        self.sub = nn.ModuleList([Sublayer(size, dropout) for _ in range(2)])

    def forward(self, x, mask=None):
        x = self.sub[0](x, lambda x: self.attn(x, mask))
        return self.sub[1](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer_size(layer))

    def forward(self, x, mask=None, return_all=False):
        outs = []
        for layer in self.layers:
            x = layer(x, mask)
            outs.append(x)
        x = self.norm(x)
        return (x, outs) if return_all else x


def layer_size(layer):
    return layer.sub[0].norm.normalized_shape[0]


# ============ 2. 构造合成信号 ============
def make_signal(B=1, L=128, d_model=64, seed=0):
    """
    输入：多个不同频率的正弦波叠加 + 噪声
    """
    torch.manual_seed(seed)
    t = torch.linspace(0, 2 * math.pi, L)
    x = torch.zeros(B, L, d_model)
    # 每个通道用不同频率（低频 + 高频混合）
    for d in range(d_model):
        freq = 1 + (d % 8) * 2          # 频率 1,3,5,...,15
        x[:, :, d] = torch.sin(freq * t).unsqueeze(0)
    x = x + 0.1 * torch.randn_like(x)   # 加噪声
    return x


# ============ 3. 傅里叶分析 ============
def spectrum(x, dim_seq=1):
    """
    x: (B, L, d)
    返回归一化平均功率谱 (L//2+1,)
    """
    x = x - x.mean(dim=dim_seq, keepdim=True)      # 去 DC
    Xf = torch.fft.rfft(x, dim=dim_seq)
    power = Xf.abs() ** 2
    ps = power.mean(dim=(0, 2))                    # 对 batch、d 平均
    return ps / ps.sum()


def high_freq_ratio(ps):
    """高频能量占比（频率高于一半的算高频）"""
    n = ps.size(0)
    return ps[n // 2:].sum().item()


# ============ 4. 主实验 ============
def run_experiment():
    torch.manual_seed(42)
    d_model, h, d_ff, N = 64, 4, 256, 4
    L = 128

    # 构造信号
    x = make_signal(B=1, L=L, d_model=d_model)

    # 构建模型
    attn = MultiHeadAttention(d_model, h)
    ff = FeedForward(d_model, d_ff)
    layer = EncoderLayer(d_model, attn, ff)
    encoder = Encoder(layer, N)
    encoder.eval()

    # 前向，收集每层输出
    with torch.no_grad():
        _, layer_outs = encoder(x, mask=None, return_all=True)

    # 分析每层频谱
    spectra = [spectrum(x)] + [spectrum(o) for o in layer_outs]
    labels = ["input"] + [f"layer {i+1}" for i in range(N)]

    print("各层高频能量占比:")
    for lbl, ps in zip(labels, spectra):
        print(f"  {lbl:>8s}: {high_freq_ratio(ps):.3f}")

    # ============ 5. 可视化 ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：每层功率谱
    freqs = torch.fft.rfftfreq(L)
    for lbl, ps in zip(labels, spectra):
        axes[0].plot(freqs.numpy(), ps.numpy(), label=lbl, alpha=0.8)
    axes[0].set_xlabel("Frequency")
    axes[0].set_ylabel("Normalized Power")
    axes[0].set_title("Spectrum per Layer")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右：高频占比柱状图
    ratios = [high_freq_ratio(ps) for ps in spectra]
    axes[1].bar(labels, ratios, color="steelblue")
    axes[1].set_ylabel("High-Freq Energy Ratio")
    axes[1].set_title("High-Frequency Energy by Layer")
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("layer_spectrum.png", dpi=120)
    plt.show()
    print("\n图已保存到 layer_spectrum.png")


if __name__ == "__main__":
    run_experiment()