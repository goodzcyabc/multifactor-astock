"""图表。统一风格,自动选择可用的中文字体。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from . import config as C

_CJK = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Songti SC",
        "Arial Unicode MS", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "Microsoft YaHei"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
_font = next((f for f in _CJK if f in _avail), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"   # 对数坐标刻度用 mathtext,中文字体缺少 U+2212 负号
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

PALETTE = ["#1f4e79", "#c0504d", "#4f9d69", "#e0a028", "#7b5ea7", "#3b9ab2", "#8c8c8c"]


def _save(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_nav(curves: dict, title: str, name: str, logy: bool = True, drawdown: pd.Series = None):
    fig, axes = plt.subplots(2 if drawdown is not None else 1, 1, figsize=(10, 6.5 if drawdown is not None else 4.5),
                             sharex=True, gridspec_kw={"height_ratios": [3, 1]} if drawdown is not None else None)
    ax = axes[0] if drawdown is not None else axes
    for i, (k, v) in enumerate(curves.items()):
        ax.plot(v.index, v.values, label=k, color=PALETTE[i % len(PALETTE)], lw=1.6 if i == 0 else 1.1)
    if logy:
        ax.set_yscale("log")
        # 对数坐标默认用 mathtext 写成 6x10^-1,中文字体缺 U+2212 会显示成乱码;改成普通数字
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v < 1 else ""))
    ax.set_title(title)
    ax.set_ylabel("净值(对数)" if logy else "净值")
    ax.legend(loc="upper left")
    if drawdown is not None:
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color=PALETTE[1], alpha=0.5)
        axes[1].set_ylabel("回撤")
    return _save(fig, name)


def plot_ic(ic: pd.Series, title: str, name: str):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = np.where(ic.values >= 0, PALETTE[2], PALETTE[1])
    ax.bar(ic.index, ic.values, width=20, color=colors, alpha=0.8, label="月度 RankIC")
    ax2 = ax.twinx()
    ax2.plot(ic.index, ic.cumsum().values, color=PALETTE[0], lw=1.5, label="累计 IC")
    ax2.grid(False)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title)
    ax.set_ylabel("RankIC"); ax2.set_ylabel("累计 IC")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    return _save(fig, name)


def plot_cum_ic_all(ics: pd.DataFrame, labels: dict, name: str):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, c in enumerate(ics.columns):
        ax.plot(ics.index, ics[c].fillna(0).cumsum(), label=labels.get(c, c), color=PALETTE[i % len(PALETTE)])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("各因子累计 RankIC(斜率 = 因子有效性,越稳定向上越好)")
    ax.legend(ncol=2)
    return _save(fig, name)


def plot_quantiles(q: pd.DataFrame, title: str, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    qcols = [c for c in q.columns if c.startswith("Q")]
    cum = (1 + q[qcols].fillna(0)).cumprod()
    for i, c in enumerate(qcols):
        axes[0].plot(cum.index, cum[c], label=c, color=plt.cm.viridis(i / max(len(qcols) - 1, 1)))
    ls = (1 + q["LS"].fillna(0)).cumprod()
    axes[0].plot(ls.index, ls, label=f"{qcols[-1]}-Q1 多空", color=PALETTE[1], ls="--")
    axes[0].set_yscale("log"); axes[0].set_title(title + ":分组累计净值"); axes[0].legend(fontsize=8)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if v < 1 else ""))
    qq = q[qcols].dropna()
    ann = (1 + qq).prod() ** (12.0 / max(len(qq), 1)) - 1.0
    axes[1].bar(qcols, ann.values, color=[plt.cm.viridis(i / max(len(qcols) - 1, 1)) for i in range(len(qcols))])
    axes[1].set_title("各组年化收益(单调性)"); axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    return _save(fig, name)


def plot_corr(corr: pd.DataFrame, labels: dict, name: str):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lab = [labels.get(c, c) for c in corr.columns]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right")
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("因子间平均截面秩相关")
    return _save(fig, name)


def plot_yearly(tab: pd.DataFrame, cols: list, name: str, title="分年收益"):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(tab)); wdt = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(x + i * wdt - 0.4 + wdt / 2, tab[c].values, width=wdt, label=c, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x); ax.set_xticklabels(tab.index)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.axhline(0, color="k", lw=0.6); ax.legend(); ax.set_title(title)
    return _save(fig, name)


def plot_series_bars(s: pd.Series, title: str, name: str, pct=True):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(s.index, s.values, width=20, color=PALETTE[0], alpha=0.8)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    return _save(fig, name)
