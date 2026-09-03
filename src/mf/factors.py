"""因子定义。每个函数输入 Market,输出"日频宽表"(值越大 = 预期未来收益越高,方向已统一)。

因子              经济逻辑(大白话)
mom_12_1          过去一年(剔除最近一个月)涨得多的股票,趟势倾向延续
rev_1m            最近一个月涨得多的股票短期倾向回落(A 股短期反转很强),取负号
vol_60            过去 60 天波动大的股票未来收益偏低(低波异象),取负号
size              流通市值小的股票长期收益偏高(A 股小盘效应),取 -log(市值)
turnover_20       过去 20 天平均换手率高的股票被过度交易、未来收益偏低,取负号
bp                账面市值比 = 1/PB,越便宜越好(价值)
roe               净资产收益率 ≈ PB/PE(TTM),盈利能力越强越好(质量)
"""
import numpy as np
import pandas as pd

from .data import Market

FACTOR_NAMES = ["mom_12_1", "rev_1m", "vol_60", "size", "turnover_20", "bp", "roe"]   # 进入合成的 7 个
DIAG_FACTORS = ["ep"]   # 只做检验、不进合成的诊断因子(Liu-Stambaugh-Yuan 2019 认为 A 股价值用 EP 优于 BP)
FACTOR_LABELS = {
    "mom_12_1": "动量(12-1月)", "rev_1m": "反转(1月)", "vol_60": "低波动(60日)",
    "size": "小市值", "turnover_20": "低换手(20日)", "bp": "价值(BP)", "roe": "质量(ROE)",
    "ep": "价值(EP,诊断)",
}


def mom_12_1(m: Market) -> pd.DataFrame:
    c = m.close_adj
    return c.shift(21) / c.shift(252) - 1.0


def rev_1m(m: Market) -> pd.DataFrame:
    c = m.close_adj
    return -(c / c.shift(21) - 1.0)


def vol_60(m: Market) -> pd.DataFrame:
    """停牌日的 0 收益不计入(否则长期停牌股票会显得"低波")。"""
    return -m.ret.where(m.trading).rolling(60, min_periods=40).std()


def size(m: Market) -> pd.DataFrame:
    return -np.log(m.float_mcap)


def turnover_20(m: Market) -> pd.DataFrame:
    t = m.turn.where(m.turn > 0)
    return -t.rolling(20, min_periods=10).mean()


def bp(m: Market) -> pd.DataFrame:
    pb = m.pb.where(m.pb > 0)
    return 1.0 / pb


def roe(m: Market) -> pd.DataFrame:
    """ROE_TTM = E_TTM / B = (P/B) / (P/E)。PE 为负(亏损)时 ROE 为负,保留;PE 为 0 视为缺失。"""
    pe = m.pe.where(m.pe != 0)
    pb = m.pb.where(m.pb > 0)
    return pb / pe


def ep(m: Market) -> pd.DataFrame:
    """盈利收益率 = 1/PE(TTM)。亏损股 PE 为负 -> EP 为负(最贵),单调性正确,保留。"""
    pe = m.pe.where(m.pe != 0)
    return 1.0 / pe


FACTOR_FUNCS = {
    "mom_12_1": mom_12_1, "rev_1m": rev_1m, "vol_60": vol_60, "size": size,
    "turnover_20": turnover_20, "bp": bp, "roe": roe, "ep": ep,
}


def compute_raw_factors(m: Market, dates: pd.DatetimeIndex) -> dict:
    """在给定信号日(通常是月末)采样所有因子的原始值。返回 {name: DataFrame(dates x codes)}。"""
    out = {}
    for name, fn in FACTOR_FUNCS.items():
        full = fn(m)
        out[name] = full.reindex(dates)
    return out
