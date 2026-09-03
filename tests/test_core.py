"""不依赖下载数据的单元测试:预处理、合成、IC 加权无前视、回测机制。运行: python3 -m pytest tests 或 python3 tests/test_core.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mf.preprocess import winsorize_mad, zscore, neutralize, process_cross_section  # noqa: E402
from mf.portfolio import composite_score, ic_ir_weights, select_top_equal  # noqa: E402
from mf.backtest import run_backtest  # noqa: E402
from mf.analysis import rank_ic, quantile_returns  # noqa: E402


def test_winsorize_and_zscore():
    x = pd.Series([1, 2, 3, 4, 5, 100.0])
    w = winsorize_mad(x)
    assert w.max() < 100 and w.idxmax() == 5
    z = zscore(w)
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9


def test_neutralize_removes_industry_mean():
    rng = np.random.default_rng(0)
    ind = pd.Series(["A"] * 50 + ["B"] * 50)
    x = pd.Series(rng.normal(size=100)) + ind.map({"A": 5.0, "B": -5.0})
    r = neutralize(x, industry=ind)
    assert abs(r[ind == "A"].mean()) < 1e-6 and abs(r[ind == "B"].mean()) < 1e-6


def test_composite_handles_nan_and_min_factors():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME"); cols = ["a", "b", "c"]
    f1 = pd.DataFrame([[1, 2, np.nan], [1, 1, 1]], index=idx, columns=cols, dtype=float)
    f2 = pd.DataFrame([[3, np.nan, np.nan], [0, 0, 0]], index=idx, columns=cols, dtype=float)
    s = composite_score({"f1": f1, "f2": f2}, min_factors=1)
    assert s.loc[idx[0], "a"] == 2.0 and s.loc[idx[0], "b"] == 2.0 and np.isnan(s.loc[idx[0], "c"])
    s2 = composite_score({"f1": f1, "f2": f2}, min_factors=2)
    assert np.isnan(s2.loc[idx[0], "b"])
    w = pd.DataFrame([[0.25, 0.75], [0.5, 0.5]], index=idx, columns=["f1", "f2"])
    s3 = composite_score({"f1": f1, "f2": f2}, weights=w, min_factors=1)
    assert abs(s3.loc[idx[0], "a"] - (0.25 * 1 + 0.75 * 3)) < 1e-12


def test_ic_ir_weights_use_only_past():
    idx = pd.date_range("2020-01-31", periods=20, freq="ME")
    ic = pd.DataFrame({"a": np.linspace(0.01, 0.2, 20), "b": np.linspace(0.2, 0.01, 20)}, index=idx)
    w = ic_ir_weights(ic, window=6, min_periods=3)
    # 改变第 10 期以后的 IC 不应影响第 10 期及以前的权重
    ic2 = ic.copy(); ic2.iloc[10:] = -1.0
    w2 = ic_ir_weights(ic2, window=6, min_periods=3)
    pd.testing.assert_frame_equal(w.iloc[:11], w2.iloc[:11])
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w.iloc[:3] == 0.5).all().all()   # 数据不足时等权


def test_select_top_equal():
    s = pd.Series({"a": 3.0, "b": np.nan, "c": 1.0, "d": 2.0})
    w = select_top_equal(s, 2)
    assert set(w.index) == {"a", "d"} and abs(w.sum() - 1) < 1e-12


def _fake_market():
    dates = pd.bdate_range("2020-01-01", periods=12)
    codes = ["x", "y", "z"]
    ret = pd.DataFrame(0.0, index=dates, columns=codes)
    ret["x"] = 0.01; ret["y"] = -0.01; ret["z"] = 0.0
    trading = pd.DataFrame(True, index=dates, columns=codes)
    lim_up = pd.DataFrame(False, index=dates, columns=codes)
    lim_dn = pd.DataFrame(False, index=dates, columns=codes)
    trading.loc[dates[6], "z"] = False           # z 在第二次执行日停牌
    lim_up.loc[dates[1], "y"] = True             # y 在第一次执行日涨停 -> 买不进
    m = SimpleNamespace(dates=dates, ret=ret, trading=trading, limit_up=lim_up, limit_down=lim_dn,
                        last_date=pd.Series(dates[-1], index=codes))
    m.next_trading_day = lambda d: dates[dates.searchsorted(d, side="right")]
    return m, dates


def test_backtest_mechanics_refined():
    m, dates = _fake_market()
    targets = {dates[0]: pd.Series({"x": 0.5, "y": 0.5}), dates[5]: pd.Series({"z": 1.0})}
    res = run_backtest(m, targets, end=dates[-1], commission=0.001, stamp_tax=0.001, slippage=0.0)
    h1 = res.holdings[dates[1]]
    assert set(h1.index) == {"x"} and abs(h1.sum() - 1) < 1e-12
    assert res.unfilled[dates[1]] == 1
    assert abs(res.cost[dates[1]] - 0.001) < 1e-12
    # 第二次执行:目标 z 停牌买不进,x 被卖出(付佣金+印花税),组合变为现金
    assert len(res.holdings[dates[6]]) == 0
    assert abs(res.cost[dates[6]] - (0.001 + 0.001)) < 1e-12
    # 之后净值不变
    assert np.allclose(res.nav.loc[dates[7]:].values, res.nav.loc[dates[6]])
    # 期间收益:第 2..6 日持有 x 每日 +1%
    expected = (1 - 0.001) * (1.01 ** 5) * (1 - 0.002)
    assert abs(res.nav.loc[dates[6]] - expected) < 1e-9


def test_frozen_position_kept_when_untradable():
    m, dates = _fake_market()
    m.trading.loc[dates[6], "x"] = False          # x 在第二次执行日停牌 -> 卖不出,冻结
    m.trading.loc[dates[6], "z"] = True
    targets = {dates[0]: pd.Series({"x": 1.0}), dates[5]: pd.Series({"z": 1.0})}
    res = run_backtest(m, targets, end=dates[-1], commission=0, stamp_tax=0, slippage=0)
    h2 = res.holdings[dates[6]]
    assert set(h2.index) == {"x"} and abs(h2["x"] - 1.0) < 1e-12 and res.turnover[dates[6]] == 0


def test_rank_ic_and_quantiles():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME"); cols = [f"c{i}" for i in range(60)]
    rng = np.random.default_rng(1)
    f = pd.DataFrame(rng.normal(size=(3, 60)), index=idx, columns=cols)
    fwd = f * 0.5 + rng.normal(scale=0.1, size=(3, 60))
    ic = rank_ic(f, fwd)
    assert (ic > 0.8).all()
    q = quantile_returns(f, fwd, 5)
    assert (q["LS"] > 0).all() and (q["Q5"] > q["Q1"]).all()


def test_stamp_tax_schedule_and_cost_mult():
    from mf import config as C
    assert C.stamp_tax_rate("2023-08-27") == 0.001 and C.stamp_tax_rate("2023-08-28") == 0.0005
    m, dates = _fake_market()
    targets = {dates[0]: pd.Series({"x": 1.0})}
    r0 = run_backtest(m, targets, end=dates[-1], cost_mult=0.0)
    r1 = run_backtest(m, targets, end=dates[-1], cost_mult=1.0)
    assert r0.cost[dates[1]] == 0 and r1.cost[dates[1]] > 0


def test_held_target_limit_up_is_kept_not_sold():
    m, dates = _fake_market()
    m.limit_up.loc[dates[6], "x"] = True            # x 已持有、仍在目标、执行日涨停 -> 不能加仓但应继续持有
    targets = {dates[0]: pd.Series({"x": 1.0}), dates[5]: pd.Series({"x": 0.5, "z": 0.5})}
    res = run_backtest(m, targets, end=dates[-1], commission=0, stamp_tax=0, slippage=0)
    h2 = res.holdings[dates[6]]
    assert "x" in h2.index and abs(h2["x"] - 1.0) < 1e-12 and "z" not in h2.index  # x 权重 100%,没有资金给 z
    assert res.unfilled[dates[6]] == 1      # z 在该日停牌(夹具设定),计 1 个未成交;x 涨停但已持有,不计


def test_nav_first_day_includes_entry_cost():
    from mf.metrics import perf_stats
    m, dates = _fake_market()
    targets = {dates[0]: pd.Series({"x": 1.0})}
    res = run_backtest(m, targets, end=dates[-1], commission=0.001, stamp_tax=0, slippage=0)
    assert res.nav.index[0] == dates[1] and abs(res.nav.iloc[0] - (1 - 0.001)) < 1e-12
    st = perf_stats(res.nav, base=1.0)
    assert abs(st["累计收益"] - (res.nav.iloc[-1] - 1)) < 1e-12       # 以初始资金 1.0 为基数,建仓成本计入


def test_delist_haircut():
    m, dates = _fake_market()
    m.last_date["x"] = dates[3]                       # x 在 dates[3] 后没有数据(退市)
    m.ret.loc[dates[4]:, "x"] = 0.0
    targets = {dates[0]: pd.Series({"x": 0.5, "z": 0.5}), dates[5]: pd.Series({"z": 1.0})}
    r0 = run_backtest(m, targets, end=dates[-1], commission=0, stamp_tax=0, slippage=0, delist_haircut=0.0)
    r1 = run_backtest(m, targets, end=dates[-1], commission=0, stamp_tax=0, slippage=0, delist_haircut=1.0)
    assert r0.delisted[dates[6]] == 1
    wx = 0.5 * 1.01 ** 2 / (0.5 * 1.01 ** 2 + 0.5)    # 退市时 x 的权重(dates[2]、dates[3] 两天各 +1%)
    assert abs(r1.nav.loc[dates[6]] / r0.nav.loc[dates[6]] - (1 - wx)) < 1e-9


def test_perf_stats_basic():
    from mf.metrics import perf_stats
    idx = pd.bdate_range("2020-01-01", periods=500)
    nav = pd.Series(np.linspace(1, 1.5, 500), index=idx)
    st = perf_stats(nav, nav)
    yrs = (idx[-1] - idx[0]).days / 365.25
    assert abs(st["年化收益"] - (1.5 ** (1 / yrs) - 1)) < 1e-12
    assert st["最大回撤"] == 0 and abs(st["年化超额收益"]) < 1e-12


def test_select_top_with_buffer():
    from mf.portfolio import select_top_with_buffer
    score = pd.Series({"a": 10, "b": 9, "c": 8, "d": 7, "e": 6, "f": 5.0})
    w0 = select_top_with_buffer(score, None, 2, 4)
    assert set(w0.index) == {"a", "b"}
    score2 = pd.Series({"a": 1, "b": 9, "c": 10, "d": 8, "e": 6, "f": 5.0})   # a 掉到第 6,b 掉到第 3
    w1 = select_top_with_buffer(score2, list(w0.index), 2, 4)
    assert set(w1.index) == {"b", "c"} and abs(w1.sum() - 1) < 1e-12          # b 在缓冲区内保留,a 换成 c
    w2 = select_top_with_buffer(score2, list(w0.index), 2, 2)
    assert set(w2.index) == {"b", "c"}                                        # 缓冲=持股数时等价于无缓冲(前 2 名正是 c、b)
    w3 = select_top_with_buffer(score2, ["a", "f"], 2, 4)
    assert set(w3.index) == {"c", "b"}                                        # 持有股都掉出缓冲区 -> 全部换成排名前 2


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