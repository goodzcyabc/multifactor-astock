"""因子合成与选股。"""
import numpy as np
import pandas as pd


def composite_score(processed: dict, weights: pd.DataFrame = None, min_factors: int = 5) -> pd.DataFrame:
    """把多个已标准化因子合成总分。
    weights: None -> 等权;否则为 DataFrame(dates x factor_names),每行为该期权重(已归一)。
    某股票某期非缺失因子数 < min_factors 时总分记 NaN(信息太少不选)。"""
    names = list(processed)
    stack = np.stack([processed[k].values for k in names], axis=-1)  # dates x codes x factors
    valid = ~np.isnan(stack)
    if weights is None:
        w = np.full((stack.shape[0], len(names)), 1.0 / len(names))
    else:
        w = weights.reindex(index=processed[names[0]].index, columns=names).fillna(0).values
    w3 = np.broadcast_to(w[:, None, :], stack.shape)
    num = np.nansum(np.where(valid, stack * w3, 0.0), axis=-1)
    den = np.where(valid, w3, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where((den > 0) & (valid.sum(-1) >= min_factors), num / den, np.nan)
    return pd.DataFrame(score, index=processed[names[0]].index, columns=processed[names[0]].columns)


def ic_ir_weights(ic: pd.DataFrame, window: int = 12, min_periods: int = 6) -> pd.DataFrame:
    """滚动 IC-IR 加权。ic 的行索引为信号日 s,度量 s->s+1 的表现,只有到 s+1 才知道,
    因此对信号日 t 可用的是 ic.shift(1) 及更早的值(严格只用过去)。负 IR 截为 0;全 0 时退回等权。"""
    past = ic.shift(1)
    mu = past.rolling(window, min_periods=min_periods).mean()
    sd = past.rolling(window, min_periods=min_periods).std()
    ir = (mu / sd).clip(lower=0).fillna(0)
    s = ir.sum(axis=1)
    w = ir.div(s.where(s > 0), axis=0)
    eq = pd.DataFrame(1.0 / ic.shape[1], index=ic.index, columns=ic.columns)
    return w.where(s > 0, eq)


def select_top_equal(score: pd.Series, n: int) -> pd.Series:
    """取总分最高的 n 只,等权。"""
    s = score.dropna().sort_values(ascending=False).head(n)
    if len(s) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(s), index=s.index)


def equal_weight(codes) -> pd.Series:
    codes = list(codes)
    return pd.Series(1.0 / len(codes), index=codes) if codes else pd.Series(dtype=float)


def select_top_with_buffer(score: pd.Series, prev, n: int, buffer_rank: int) -> pd.Series:
    """换仓缓冲:上期持有、且本期排名仍在前 buffer_rank 名内的股票继续持有;
    空出的名额按排名从其余股票中补足。用于降低换手,是组合层面的执行规则,不改变因子。"""
    s = score.dropna().sort_values(ascending=False)
    if len(s) == 0:
        return pd.Series(dtype=float)
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    prev = [c for c in (prev if prev is not None else []) if c in rank.index]
    keep = [c for c in prev if rank[c] <= buffer_rank][:n]
    need = n - len(keep)
    fill = [c for c in s.index if c not in keep][:max(need, 0)]
    sel = keep + fill
    return pd.Series(1.0 / len(sel), index=sel)
