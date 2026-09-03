"""绩效指标。"""
import numpy as np
import pandas as pd

ANN = 243   # A 股每年约 243 个交易日(2016-2025 平均),用于波动率年化


def _years(idx: pd.DatetimeIndex) -> float:
    """按日历时间计算年数,避免交易日数与 252 假设不一致。"""
    return (idx[-1] - idx[0]).days / 365.25


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def perf_stats(nav: pd.Series, bench: pd.Series = None, rf: float = 0.0, base: float = None) -> dict:
    """base: 初始资金基数;None 表示用 nav 首值(区间相对表现)。全样本策略传 base=1.0,使首日建仓成本计入收益。"""
    nav = nav.dropna()
    base = nav.iloc[0] if base is None else base
    r = nav.pct_change().dropna()
    n_years = _years(nav.index)
    ann_ret = (nav.iloc[-1] / base) ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(ANN)
    dd = drawdown(nav)
    mdd = dd.min()
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    out = {
        "年化收益": ann_ret, "年化波动": ann_vol,
        "夏普比率": (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan,
        "最大回撤": mdd, "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "最大回撤起止": f"{peak:%Y-%m-%d}~{trough:%Y-%m-%d}",
        "累计收益": nav.iloc[-1] / base - 1.0,
    }
    if bench is not None:
        b = bench.reindex(nav.index).ffill()
        b = b / b.iloc[0]
        rb = b.pct_change().dropna()
        rel = (nav / base) / b
        ann_b = (b.iloc[-1] / b.iloc[0]) ** (1.0 / n_years) - 1.0
        ex = r - rb.reindex(r.index).fillna(0)
        te = ex.std() * np.sqrt(ANN)
        ann_ex = rel.iloc[-1] ** (1.0 / n_years) - 1.0
        beta = np.cov(r, rb.reindex(r.index).fillna(0))[0, 1] / rb.var() if rb.var() > 0 else np.nan
        mr = nav.resample("ME").last().pct_change().dropna()
        mb = b.resample("ME").last().pct_change().dropna()
        out.update({
            "基准年化收益": ann_b, "年化超额收益": ann_ex, "跟踪误差": te,
            "信息比率": ann_ex / te if te > 0 else np.nan, "Beta": beta,
            "超额最大回撤": drawdown(rel).min(), "月度胜率(vs基准)": (mr > mb.reindex(mr.index)).mean(),
        })
    return out


def yearly_table(nav: pd.Series, benches: dict, nav0: float = None) -> pd.DataFrame:
    """分年收益:策略、各基准、相对第一个基准的超额、策略年内最大回撤。
    nav0: 策略首年的基数(传 1.0 可让首年包含建仓成本);None 表示用当年首值。"""
    def yr_ret(s, base0=None):
        s = s.dropna()
        first = s.groupby(s.index.year).first()
        last = s.groupby(s.index.year).last()
        prev_last = last.shift(1)
        base = prev_last.fillna(first if base0 is None else base0)
        return last / base - 1.0
    tab = pd.DataFrame({"策略": yr_ret(nav, nav0)})
    for k, v in benches.items():
        tab[k] = yr_ret(v.reindex(nav.index).ffill())
    first_b = list(benches)[0]
    tab[f"超额(vs{first_b})"] = (1 + tab["策略"]) / (1 + tab[first_b]) - 1
    tab["策略年内最大回撤"] = nav.groupby(nav.index.year).apply(lambda s: drawdown(s).min())
    return tab


def fmt_pct(x, nd=1):
    return "-" if pd.isna(x) else f"{x*100:.{nd}f}%"
