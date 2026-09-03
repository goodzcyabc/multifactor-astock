"""原始数据加载、宽表构建(Market 对象)。

约定:
- 所有宽表 index 为交易日 DatetimeIndex,columns 为 baostock 代码(如 sh.600000)。
- 价格为后复权价,收益率直接用 close.pct_change()。
- 流通市值 = 成交额 / (换手率/100)。恒等式来源:换手率 = 成交量/流通股本,成交额 = Σ价格×成交量,
  因此 成交额/(换手率) = 流通股本 × 当日成交均价 ≈ 流通市值。停牌日(换手为 0)取 NaN 后向前填充。
"""
import json
import re
import numpy as np
import pandas as pd

from . import config as C

NUM_COLS = ["close", "amount", "turn", "pctChg", "peTTM", "pbMRQ"]


def load_trade_dates() -> pd.DatetimeIndex:
    td = pd.to_datetime(pd.read_csv(C.DATA_RAW / "trade_dates.csv")["date"])
    return pd.DatetimeIndex(sorted(td.unique()))


def load_constituents() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_RAW / "constituents.csv", dtype={"code": str})
    df["asof"] = pd.to_datetime(df["asof"])
    return df[["asof", "code", "index", "code_name"]]


def load_index(code: str) -> pd.DataFrame:
    df = pd.read_csv(C.DATA_RAW / f"index_{code}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    return df


def load_industry() -> pd.Series:
    """证监会行业分类 -> 行业组。制造业(C)细分到 2 位大类,其余用门类字母,避免组太大或太小。
    注意:这是最新快照,已退市股票没有分类 -> NaN。中性化时 NaN 不单独成组(否则"无行业"组
    几乎全是未来退市股,等于泄露未来),而是只对市值中性、行业效应按平均处理。"""
    df = pd.read_csv(C.DATA_RAW / "industry.csv", dtype=str)
    def grp(s):
        if not isinstance(s, str):
            return np.nan
        m = re.match(r"^([A-Z])(\d{2})", s)
        if not m:
            return np.nan
        return m.group(1) + m.group(2) if m.group(1) == "C" else m.group(1)
    return pd.Series(df["industry"].map(grp).values, index=df["code"].values, name="industry")


def load_stock_basic() -> pd.DataFrame:
    f = C.DATA_RAW / "stock_basic.csv"
    empty = pd.DataFrame(columns=["code", "ipoDate", "outDate"]).set_index("code")
    if not f.exists() or f.stat().st_size == 0:
        return empty
    try:
        df = pd.read_csv(f, dtype=str)
    except pd.errors.EmptyDataError:
        return empty
    if "code" not in df.columns:
        return empty
    df = df.drop_duplicates("code").set_index("code")
    df["ipoDate"] = pd.to_datetime(df["ipoDate"], errors="coerce")
    df["outDate"] = pd.to_datetime(df["outDate"], errors="coerce")
    return df


def build_long(force: bool = False) -> pd.DataFrame:
    """把 bars/*.csv 合并成长表并缓存为 parquet。"""
    cache = C.DATA_PROC / "bars_long.parquet"
    manifest_f = C.DATA_PROC / "bars_long.manifest.json"
    files = sorted(C.DATA_BARS.glob("*.csv"))
    manifest = {"n_files": len(files), "max_mtime": max((f.stat().st_mtime for f in files), default=0),
                "total_size": sum(f.stat().st_size for f in files)}
    if cache.exists() and not force and manifest_f.exists():
        try:
            if json.load(open(manifest_f)) == manifest:
                return pd.read_parquet(cache)
        except Exception:  # noqa: BLE001
            pass
    # 缓存不存在、或 bars/ 目录有变化(文件数/大小/修改时间) -> 重建,避免静默使用不完整的数据
    frames = []
    for f in sorted(C.DATA_BARS.glob("*.csv")):
        d = pd.read_csv(f, dtype=str)
        if len(d):
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype(int)
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df.to_parquet(cache)
    json.dump(manifest, open(manifest_f, "w"))
    return df


def _wide(long: pd.DataFrame, col: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return long.pivot(index="date", columns="code", values=col).reindex(dates)


def board_limit_pct(codes, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """每只股票每天的涨跌幅限制(%),不含 ST 的 5% 规则(在 Market 中叠加)。
    主板 10%;科创板(688)20%;创业板(30xxxx)2020-08-24 起 20%,之前 10%;北交所 30%。"""
    lim = pd.DataFrame(10.0, index=dates, columns=codes)
    for c in codes:
        num = c.split(".")[-1]
        if num.startswith("688") or num.startswith("689"):
            lim[c] = 20.0
        elif num.startswith("30"):
            lim.loc[lim.index >= "2020-08-24", c] = 20.0
        elif num.startswith(("4", "8")):
            lim[c] = 30.0
    return lim


class Market:
    """把所有宽表装在一起,供因子和回测使用。"""

    def __init__(self, force: bool = False):
        long = build_long(force=force)
        self.dates = load_trade_dates()
        last_bar = long["date"].max()
        try:
            last_bar = min(last_bar, load_index(C.BENCHMARK_CODE).index.max())
        except FileNotFoundError:
            pass
        # baostock 的交易日历对尚未发生的日期也标 is_trading_day=1,因此以实际有行情的最后一天为界
        self.dates = self.dates[(self.dates >= C.DATA_START) & (self.dates <= min(pd.Timestamp(C.DATA_END), last_bar))]
        self.codes = sorted(long["code"].unique())

        self.close = _wide(long, "close", self.dates)
        self.amount = _wide(long, "amount", self.dates)
        self.turn = _wide(long, "turn", self.dates)
        self.pct = _wide(long, "pctChg", self.dates)
        self.pe = _wide(long, "peTTM", self.dates)
        self.pb = _wide(long, "pbMRQ", self.dates)
        self.is_st = _wide(long, "isST", self.dates).fillna(0).astype(bool)
        # 当日真正可成交:交易状态正常且有成交额
        self.trading = (_wide(long, "tradestatus", self.dates).fillna(0) == 1) & (self.amount.fillna(0) > 0)

        # 有行情记录的第一天/最后一天(用于新股过滤和退市处理)
        self.first_date = self.close.apply(lambda s: s.first_valid_index())
        self.last_date = self.close.apply(lambda s: s.last_valid_index())
        basic = load_stock_basic()
        ipo = basic["ipoDate"].reindex(self.codes) if len(basic) else pd.Series(pd.NaT, index=self.codes)
        self.ipo_date = ipo.fillna(self.first_date)

        # ---- 收益率与一致的复权价格
        # baostock 的后复权收盘价偶发永久性错位(如 sz.000001 2020-12-31 复权价 -16.2%,官方涨跌幅 +0.73%),
        # 官方涨跌幅 pctChg 也偶有 0 占位(如 sh.689009 2020-11-02)。两者相差 >1 个百分点时,用 PB 的日变化做裁判
        # (PB = 原始价/每股净资产,随原始价同步变动),谁离 PB 的变动更近就信谁;再用累计修正系数重建一条一致的
        # 复权价 close_adj,供收益率、动量/反转等价格比率因子和前向收益共用。修正只用 <=t 的数据(cumprod),无前视。
        close_ff = self.close.ffill()
        r_adj = close_ff.pct_change()
        r_pct = self.pct / 100.0
        r_pb = self.pb.ffill().pct_change()
        disagree = ((r_adj - r_pct).abs() > 0.01) & r_adj.notna() & r_pct.notna()
        use_pct = disagree & ((r_pct - r_pb).abs() < (r_adj - r_pb).abs())
        self.repaired = use_pct   # 被官方涨跌幅替换的 (日, 股票) 单元格
        corr = ((1.0 + r_pct) / (1.0 + r_adj)).where(use_pct, 1.0).fillna(1.0).cumprod()
        self.close_adj = (close_ff * corr).where(close_ff.notna())
        # 日收益:停牌日 close 延续前值 -> 收益 0;上市前 NaN;退市/数据结束后 NaN
        self.ret = self.close_adj.pct_change().where(close_ff.notna())

        # 流通市值(元)
        mcap = (self.amount / (self.turn / 100.0)).where(self.turn > 0)
        self.float_mcap = mcap.ffill(limit=60)

        # 涨跌停判定(按收盘涨跌幅是否贴近限制)
        lim = board_limit_pct(self.codes, self.dates)
        lim = lim.where(~(self.is_st & (lim == 10.0)), 5.0)   # ST 5% 只适用于主板;创业板/科创板 ST 仍为 20%
        self.limit_up = self.pct >= (lim - 0.5)
        self.limit_down = self.pct <= -(lim - 0.5)

    def month_ends(self, start=None, end=None) -> pd.DatetimeIndex:
        d = self.dates
        if start is not None:
            d = d[d >= pd.Timestamp(start)]
        if end is not None:
            d = d[d <= pd.Timestamp(end)]
        s = pd.Series(d, index=d)
        return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max().values)

    def next_trading_day(self, d: pd.Timestamp) -> pd.Timestamp:
        i = self.dates.searchsorted(d, side="right")
        return self.dates[i] if i < len(self.dates) else None


def universe_at(cons: pd.DataFrame, asof: pd.Timestamp) -> list:
    """取 asof 当日(或之前最近一次)的成分股名单。"""
    avail = cons["asof"].unique()
    avail = avail[avail <= asof]
    if len(avail) == 0:
        return []
    d = max(avail)
    return sorted(cons.loc[cons["asof"] == d, "code"].unique())
