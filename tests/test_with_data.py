"""依赖已下载数据的测试(数据不存在时自动跳过)。运行: python3 tests/test_with_data.py

核心是"扰动未来数据"测试:把信号日 t 之后的所有价格/成交数据随机打乱,t 日的因子分和股票池必须完全不变。
这是对"没有前视偏差"最直接的证明。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mf import config as C  # noqa: E402

HAVE_DATA = (C.DATA_RAW / "constituents.csv").exists() and any(C.DATA_BARS.glob("*.csv"))


def _market():
    from mf.data import Market
    return Market()


def test_no_lookahead_by_perturbing_future():
    if not HAVE_DATA:
        print("SKIP (no data)"); return
    import copy
    from mf.data import load_constituents
    from mf.factors import compute_raw_factors
    from mf.pipeline import build_universe_mask, process_factors
    from mf.portfolio import composite_score, select_top_equal
    m = _market()
    cons = load_constituents()
    t = m.month_ends(start="2019-01-01", end="2019-12-31")[5]        # 2019-06 月末
    dates = pd.DatetimeIndex([t])
    mask, _ = build_universe_mask(m, cons, dates)
    base = composite_score(process_factors(m, compute_raw_factors(m, dates), mask)).loc[t]
    picks = select_top_equal(base, C.TOP_N)

    m2 = copy.copy(m)
    rng = np.random.default_rng(0)
    fut = m.dates > t
    for attr in ("close", "amount", "turn", "pct", "pe", "pb"):
        df = getattr(m, attr).copy()
        noise = rng.uniform(0.5, 1.5, size=df.loc[fut].shape)
        df.loc[fut] = df.loc[fut].values * noise
        setattr(m2, attr, df)
    m2.is_st = m.is_st.copy(); m2.is_st.loc[fut] = ~m.is_st.loc[fut]
    m2.trading = m.trading.copy(); m2.trading.loc[fut] = False
    m2.close_adj = m.close_adj.copy(); m2.close_adj.loc[fut] = m2.close_adj.loc[fut].values * rng.uniform(0.5, 1.5, size=m2.close_adj.loc[fut].shape)
    m2.ret = m2.close_adj.pct_change().where(m2.close_adj.notna())
    m2.float_mcap = (m2.amount / (m2.turn / 100.0)).where(m2.turn > 0).ffill(limit=60)
    mask2, _ = build_universe_mask(m2, cons, dates)
    pert = composite_score(process_factors(m2, compute_raw_factors(m2, dates), mask2)).loc[t]
    picks2 = select_top_equal(pert, C.TOP_N)
    pd.testing.assert_series_equal(base, pert)
    assert list(picks.index) == list(picks2.index)
    print(f"  perturbed all data after {t.date()}: {len(picks)} picks unchanged")


def test_pctchg_consistent_with_adjusted_close():
    if not HAVE_DATA:
        print("SKIP (no data)"); return
    m = _market()
    both = m.trading & m.trading.shift(1).fillna(False)              # 连续两个交易日
    raw_diff = (m.close.ffill().pct_change() - m.pct / 100.0).where(both).abs()
    fixed_diff = (m.ret - m.pct / 100.0).where(both).abs()
    n = int(both.sum().sum())
    raw_bad, fixed_bad = int((raw_diff > 0.01).sum().sum()), int((fixed_diff > 0.01).sum().sum())
    print(f"  |adj-close return - pctChg| > 1pp: raw {raw_bad}/{n}, after reconciliation {fixed_bad}/{n}; "
          f"repaired cells {int(m.repaired.sum().sum())}, max remaining gap {fixed_diff.max().max():.4f}")
    assert fixed_bad <= raw_bad, "修正后反而更差"
    assert raw_bad / n < 1e-4, "原始不一致过多,数据源可能有系统性问题"
    # 修正后剩余的分歧必须要么很小(<5%,多为 PB 前值填充导致裁判失灵的边缘情形),
    # 要么是官方涨跌幅本身为 0 占位、复权价被判定正确的情形(如 sh.689009 2020-11-02)。
    remaining = fixed_diff[fixed_diff > 0.05].stack()
    assert all(m.pct.loc[d, c] == 0 for d, c in remaining.index), remaining
    assert fixed_bad < 10, f"修正后仍有 {fixed_bad} 处 >1pp 分歧"


def test_limit_bands():
    from mf.data import board_limit_pct
    dates = pd.DatetimeIndex(["2020-08-21", "2020-08-24"])
    lim = board_limit_pct(["sz.300001", "sh.688001", "sh.600000", "sz.000001"], dates)
    assert lim.loc["2020-08-21", "sz.300001"] == 10 and lim.loc["2020-08-24", "sz.300001"] == 20
    assert (lim["sh.688001"] == 20).all() and (lim["sh.600000"] == 10).all()


def test_universe_is_point_in_time():
    if not HAVE_DATA:
        print("SKIP (no data)"); return
    from mf.data import load_constituents, universe_at
    cons = load_constituents()
    early = set(universe_at(cons, pd.Timestamp("2016-01-29")))
    late = set(universe_at(cons, cons["asof"].max()))
    assert len(early) == 800 and len(late) == 800
    assert len(early - late) > 100, "名单几乎不变,疑似不是时点成分股"
    print(f"  2016-01 vs latest: {len(early - late)} names dropped, {len(late - early)} added")


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception:
                fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(1 if fails else 0)
