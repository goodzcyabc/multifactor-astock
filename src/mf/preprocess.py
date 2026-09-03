"""横截面预处理:去极值、标准化、中性化。所有函数作用在"一个截面"(Series)或宽表的每一行。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def winsorize_mad(x: pd.Series, n: float = 3.0) -> pd.Series:
    """MAD 去极值:把偏离中位数超过 n 倍 MAD 的值截断到边界。"""
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return x
    return x.clip(med - n * mad, med + n * mad)


def zscore(x: pd.Series) -> pd.Series:
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return x * 0.0
    return (x - x.mean()) / sd


def neutralize(x: pd.Series, industry: pd.Series = None, log_size: pd.Series = None) -> pd.Series:
    """对 [常数, 对数市值, 行业哑变量] 做截面 OLS 取残差。
    行业缺失的股票(退市股在最新快照中没有分类)不单独成组:其哑变量全为 0,等价于只做市值中性、
    行业效应取平均。系数用 pinv 求解,哑变量与常数共线时残差仍唯一。"""
    df = pd.DataFrame({"y": x})
    X = pd.DataFrame({"const": 1.0}, index=x.index)
    if log_size is not None:
        X["size"] = log_size.reindex(x.index)
    if industry is not None:
        ind = industry.reindex(x.index)
        X = pd.concat([X, pd.get_dummies(ind, prefix="ind", dtype=float)], axis=1)
    both = pd.concat([df, X], axis=1).dropna()
    if len(both) < 30:
        return x * np.nan
    Xm, y = both.drop(columns="y"), both["y"]
    Xm = Xm.loc[:, (Xm != 0).any()]
    res = sm.OLS(y, Xm).fit()
    out = pd.Series(np.nan, index=x.index)
    out.loc[both.index] = res.resid
    return out


def process_cross_section(x: pd.Series, industry=None, log_size=None,
                          do_neutralize: bool = False) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 30:
        return x * np.nan
    x = winsorize_mad(x)
    if do_neutralize:
        x = neutralize(x, industry=industry, log_size=log_size)
        x = x.dropna()
    return zscore(x)
