"""单因子检验:RankIC、分层回测、因子相关性、因子稳定性。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import Market


def forward_returns(m: Market, signal_dates: pd.DatetimeIndex, k: int = 1) -> pd.DataFrame:
    """信号日 t 收盘 -> 往后第 k 个信号日收盘 的个股收益。k=1 与月度持有期一致且不重叠;k>1 用于 IC 衰减。最后 k 期为 NaN。"""
    c = m.close_adj.reindex(signal_dates)
    return c.shift(-k) / c - 1.0


def forward_returns_exec(m: Market, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """与回测执行口径完全一致的前向收益:信号日 T 的下一交易日收盘 -> 下个信号日的下一交易日收盘。"""
    exec_dates = pd.DatetimeIndex([m.next_trading_day(d) for d in signal_dates if m.next_trading_day(d) is not None])
    c = m.close_adj.reindex(exec_dates)
    fwd = (c.shift(-1) / c - 1.0)
    fwd.index = signal_dates[:len(exec_dates)]
    return fwd.reindex(signal_dates)


def nw_tstat(ic: pd.Series, lags: int = 3) -> float:
    """Newey-West(HAC)修正的 IC 均值 t 值。"""
    x = ic.dropna().values
    if len(x) < 10:
        return np.nan
    res = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_n: int = 30) -> pd.Series:
    """逐期 Spearman 秩相关(RankIC)。"""
    out = {}
    for d in factor.index:
        if d not in fwd.index:
            continue
        both = pd.concat([factor.loc[d], fwd.loc[d]], axis=1).dropna()
        out[d] = both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman") if len(both) >= min_n else np.nan
    return pd.Series(out, name="ic")


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    ir = mean / std if std > 0 else np.nan
    return {
        "IC均值": mean, "IC标准差": std, "ICIR": ir, "t值": ir * np.sqrt(n) if n else np.nan,
        "t值(NW)": nw_tstat(ic), "IC>0占比": (ic > 0).mean(), "期数": n,
    }


def quantile_returns(factor: pd.DataFrame, fwd: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """按因子值分 n 组(Q1 最低 … Qn 最高),各组等权,返回每期收益;附 LS = Qn - Q1。"""
    rows = {}
    for d in factor.index:
        if d not in fwd.index:
            continue
        both = pd.concat([factor.loc[d].rename("f"), fwd.loc[d].rename("r")], axis=1).dropna()
        if len(both) < n * 10:
            continue
        both["q"] = pd.qcut(both["f"].rank(method="first"), n, labels=[f"Q{i+1}" for i in range(n)])
        g = both.groupby("q", observed=True)["r"].mean()
        rows[d] = g
    cols = [f"Q{i+1}" for i in range(n)]
    q = pd.DataFrame(rows).T.reindex(columns=cols) if rows else pd.DataFrame(columns=cols, dtype=float)
    q["LS"] = q[f"Q{n}"] - q["Q1"]
    return q


def factor_corr(processed: dict) -> pd.DataFrame:
    """各因子(已标准化)之间的平均截面 Spearman 相关。"""
    names = list(processed)
    dates = processed[names[0]].index
    mats = []
    for d in dates:
        df = pd.DataFrame({k: processed[k].loc[d] for k in names}).dropna(how="all")
        if len(df.dropna()) < 50:
            continue
        mats.append(df.corr(method="spearman"))
    return sum(mats) / len(mats) if mats else pd.DataFrame(index=names, columns=names)


def factor_autocorr(factor: pd.DataFrame) -> pd.Series:
    """相邻两期因子值的秩相关:越高说明因子越稳定、换手越低。"""
    out = {}
    idx = factor.index
    for i in range(1, len(idx)):
        a, b = factor.loc[idx[i - 1]], factor.loc[idx[i]]
        both = pd.concat([a, b], axis=1).dropna()
        out[idx[i]] = both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman") if len(both) >= 30 else np.nan
    return pd.Series(out)


def ic_decay(factor: pd.DataFrame, m: Market, dates: pd.DatetimeIndex, mask: pd.DataFrame,
             horizons=(1, 2, 3, 6, 12)) -> pd.Series:
    """因子对未来第 k 个月(非累计)收益的平均 RankIC:看预测力衰减多快。"""
    c = m.close_adj.reindex(dates)
    out = {}
    for k in horizons:
        r_k = (c.shift(-k) / c.shift(-(k - 1)) - 1.0).where(mask)   # 第 k 个月单月收益
        out[f"第{k}月"] = rank_ic(factor, r_k).mean()
    return pd.Series(out)
