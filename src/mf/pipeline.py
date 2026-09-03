"""端到端流程:股票池 -> 因子 -> 单因子检验 -> 合成 -> 回测 -> 图表与结果表。

所有中间结果保存在 report/results/,图保存在 report/figures/。
"""
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .data import Market, load_constituents, load_index, load_industry, universe_at
from .factors import compute_raw_factors, FACTOR_NAMES, DIAG_FACTORS, FACTOR_LABELS
from .preprocess import process_cross_section
from .analysis import (forward_returns, forward_returns_exec, rank_ic, ic_summary, quantile_returns,
                       factor_corr, factor_autocorr, ic_decay)
from .portfolio import composite_score, ic_ir_weights, select_top_equal, equal_weight, select_top_with_buffer
from .backtest import run_backtest
from .metrics import perf_stats, yearly_table, drawdown
from . import plotting as pl

RES = C.REPORT_DIR / "results"
RES.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- 股票池
def signal_dates(m: Market) -> pd.DatetimeIndex:
    """信号日 = 月末交易日。从回测起点前一个月末开始(它决定第一个月的持仓)。"""
    me = m.month_ends(end=C.BT_END)
    start = pd.Timestamp(C.BT_START) - pd.offsets.MonthBegin(1)
    return me[me >= start]


def build_universe_mask(m: Market, cons: pd.DataFrame, dates: pd.DatetimeIndex):
    """每个信号日的可投股票池:时点成分股,剔除 ST、当日停牌、上市不满 MIN_LISTED_DAYS 个交易日。"""
    member = pd.DataFrame(False, index=dates, columns=m.codes)
    n_list = {}
    for d in dates:
        full = universe_at(cons, d)
        n_list[d] = len(full)
        u = [c for c in full if c in member.columns]
        member.loc[d, u] = True
    st = m.is_st.reindex(dates).fillna(False)
    trading = m.trading.reindex(dates).fillna(False)
    pos = pd.Series(np.arange(len(m.dates)), index=m.dates)
    ipo_pos = pd.Series([m.dates.searchsorted(x) if pd.notna(x) else np.nan for x in m.ipo_date.reindex(m.codes)],
                        index=m.codes)
    listed = pos.reindex(dates).values[:, None] - ipo_pos.values[None, :]
    is_new = pd.DataFrame(listed < C.MIN_LISTED_DAYS, index=dates, columns=m.codes)
    ok = member & ~st & trading & ~is_new
    stats = pd.DataFrame({
        "名单股数": pd.Series(n_list), "有行情数据": member.sum(1), "剔除ST": (member & st).sum(1),
        "剔除停牌": (member & ~st & ~trading).sum(1), "剔除新股": (member & ~st & trading & is_new).sum(1),
        "最终股票池": ok.sum(1),
    })
    return ok, stats


# ---------------------------------------------------------------- 因子处理
def process_factors(m: Market, raw: dict, mask: pd.DataFrame, neutralize: bool = False,
                    industry: pd.Series = None) -> dict:
    log_size = np.log(m.float_mcap).reindex(mask.index)
    out = {}
    for name, df in raw.items():
        df = df.where(mask)
        rows = {}
        for d in df.index:
            rows[d] = process_cross_section(
                df.loc[d], industry=industry,
                log_size=None if (not neutralize or name == "size") else log_size.loc[d],
                do_neutralize=neutralize)
        out[name] = pd.DataFrame(rows).T.reindex(columns=df.columns)
    return out


# ---------------------------------------------------------------- 主流程
def run(force: bool = False, neutral_variant: bool = True):
    t0 = time.time()
    m = Market(force=force)
    log(f"market loaded: {len(m.codes)} codes, {len(m.dates)} days")
    cons = load_constituents()
    dates = signal_dates(m)
    mask, ustats = build_universe_mask(m, cons, dates)
    ustats.to_csv(RES / "universe_stats.csv")
    log(f"universe: {len(dates)} signal dates, avg pool size {ustats['最终股票池'].mean():.0f}")

    raw = compute_raw_factors(m, dates)
    industry = load_industry()
    proc = process_factors(m, raw, mask)
    log("factors processed (raw z-score)")
    fwd = forward_returns(m, dates).where(mask)
    ALL = FACTOR_NAMES + DIAG_FACTORS

    # ---- 单因子检验(合成因子 + 诊断因子)
    ic = pd.DataFrame({k: rank_ic(proc[k], fwd) for k in ALL})
    ic.to_csv(RES / "ic_monthly.csv")
    fwd_exec = forward_returns_exec(m, dates).where(mask)          # 与回测执行口径一致(T+1 收盘到下月 T+1 收盘)
    ic_exec = pd.DataFrame({k: rank_ic(proc[k], fwd_exec) for k in ALL})
    decay = pd.DataFrame({FACTOR_LABELS[k]: ic_decay(proc[k], m, dates, mask) for k in ALL}).T
    decay.to_csv(RES / "ic_decay.csv")
    summ = pd.DataFrame({k: ic_summary(ic[k]) for k in ALL}).T
    summ["IC均值(执行口径)"] = ic_exec.mean()
    summ["因子自相关"] = pd.Series({k: factor_autocorr(proc[k]).mean() for k in ALL})
    summ["缺失率"] = pd.Series({k: 1 - proc[k].notna().sum(1).div(mask.sum(1)).mean() for k in ALL})
    qs = {k: quantile_returns(proc[k], fwd, C.N_QUANTILES) for k in ALL}
    def _ann(r):   # 月度收益序列 -> 复利年化,与组合的 CAGR 口径一致
        r = r.dropna()
        return (1 + r).prod() ** (12.0 / len(r)) - 1.0 if len(r) else np.nan
    summ["多空年化(Q5-Q1)"] = pd.Series({k: _ann(qs[k]["LS"]) for k in ALL})
    summ["Q5年化"] = pd.Series({k: _ann(qs[k][f"Q{C.N_QUANTILES}"]) for k in ALL})
    summ["Q1年化"] = pd.Series({k: _ann(qs[k]["Q1"]) for k in ALL})
    summ["单调性(组均值秩相关)"] = pd.Series({
        k: pd.Series(qs[k][[f"Q{i+1}" for i in range(C.N_QUANTILES)]].mean().values).corr(
            pd.Series(range(C.N_QUANTILES)), method="spearman") for k in ALL})
    # 分段 IC
    for lab, (a, b) in {"IC均值(16-21)": ("2016", "2021"), "IC均值(22-26)": ("2022", "2027")}.items():
        summ[lab] = ic.loc[a:b].mean()
    summ.index = [FACTOR_LABELS[k] for k in summ.index]
    summ.to_csv(RES / "factor_summary.csv")
    corr = factor_corr({k: proc[k] for k in FACTOR_NAMES})
    corr.to_csv(RES / "factor_corr.csv")
    # 每个因子对 log 流通市值的截面秩相关(市值是 A 股其它因子的"底色")
    lsz = (-raw["size"]).where(mask)
    size_corr = pd.Series({FACTOR_LABELS[k]: pd.Series({d: proc[k].loc[d].corr(lsz.loc[d], method="spearman")
                                                        for d in dates}).mean() for k in ALL}, name="与log市值秩相关")
    size_corr.to_csv(RES / "factor_size_corr.csv")
    log("single-factor tests done")

    proc_n = None
    if neutral_variant:
        proc_n = process_factors(m, {k: raw[k] for k in ALL}, mask, neutralize=True, industry=industry)
        ic_n = pd.DataFrame({k: rank_ic(proc_n[k], fwd) for k in ALL})
        ic_n.to_csv(RES / "ic_monthly_neutral.csv")
        summ_n = pd.DataFrame({k: ic_summary(ic_n[k]) for k in ALL}).T
        summ_n["IC均值(中性化前)"] = ic.mean()
        summ_n.index = [FACTOR_LABELS[k] for k in summ_n.index]
        summ_n.to_csv(RES / "factor_summary_neutral.csv")
        log("neutralized variant done")

    # ---- 合成(只用 7 个预注册因子;诊断因子不进合成)
    proc7 = {k: proc[k] for k in FACTOR_NAMES}
    ic7 = ic[FACTOR_NAMES]
    scores = {"等权合成": composite_score(proc7), "IC-IR加权": composite_score(proc7, ic_ir_weights(ic7))}
    if proc_n is not None:
        # 中性化版本的目的就是剔除市值暴露,因此合成中不再放市值因子(否则市值暴露反而被放大)
        scores["行业市值中性(6因子)"] = composite_score({k: proc_n[k] for k in FACTOR_NAMES if k != "size"}, min_factors=4)
    ustats["有效总分股数"] = scores["等权合成"].notna().sum(1)
    ustats.to_csv(RES / "universe_stats.csv")
    comp_ic = pd.DataFrame({k: rank_ic(v, fwd) for k, v in scores.items()})
    comp_ic.to_csv(RES / "ic_composite.csv")
    comp_summ = pd.DataFrame({k: ic_summary(comp_ic[k]) for k in comp_ic}).T
    comp_summ.to_csv(RES / "composite_summary.csv")
    comp_q = quantile_returns(scores["等权合成"], fwd, C.N_QUANTILES)

    # ---- 回测
    def targets_from(score: pd.DataFrame, n=C.TOP_N):
        return {d: select_top_equal(score.loc[d], n) for d in score.index}

    def targets_buffer(score: pd.DataFrame, n=C.TOP_N, buffer_rank=C.BUFFER_RANK):
        out, prev = {}, None
        for d in score.index:
            out[d] = select_top_with_buffer(score.loc[d], prev, n, buffer_rank)
            prev = list(out[d].index)
        return out

    bench_px = load_index(C.BENCHMARK_CODE)["close"]
    bts = {k: run_backtest(m, targets_from(v)) for k, v in scores.items()}
    bts[f"等权合成+换仓缓冲(前{C.BUFFER_RANK}名不卖)"] = run_backtest(m, targets_buffer(scores["等权合成"]))
    bts["股票池等权"] = run_backtest(m, {d: equal_weight(mask.columns[mask.loc[d].values]) for d in dates})
    # 股票池流通市值加权、零成本、含股息(后复权价) -> 中证 800 全收益指数的近似,消除"价格指数不含股息"的高估
    mcap_w = {}
    for d in dates:
        mc = m.float_mcap.loc[d].where(mask.loc[d]).dropna()
        mcap_w[d] = mc / mc.sum()
    bts["股票池市值加权(含股息)"] = run_backtest(m, mcap_w, cost_mult=0.0)
    log("main backtests done")
    single = {FACTOR_LABELS[k]: run_backtest(m, targets_from(proc[k])) for k in ALL}
    log("single-factor backtests done")

    main = bts["等权合成"]
    idx = main.nav.index
    bench = bench_px.reindex(idx).ffill(); bench = bench / bench.iloc[0]
    ew = bts["股票池等权"].nav

    rows = {}
    for k, v in bts.items():
        rows[k] = perf_stats(v.nav, bench, base=1.0)
        # 年化单边换手:不含首次建仓,总换手 / 日历年数
        yrs = (v.nav.index[-1] - v.nav.index[0]).days / 365.25
        rows[k]["年均单边换手"] = v.turnover.iloc[1:].sum() / yrs
        rows[k]["平均持股数"] = v.n_hold.mean()
    rows[C.BENCHMARK_NAME] = perf_stats(bench, bench)
    perf = pd.DataFrame(rows).T
    perf.to_csv(RES / "performance.csv")
    perf_single = pd.DataFrame({k: perf_stats(v.nav, bench, base=1.0) for k, v in single.items()}).T
    perf_single.to_csv(RES / "performance_single_factor.csv")

    # 相对股票池等权的超额(消除等权 vs 市值加权的差异)
    rows_ew = {k: perf_stats(v.nav, ew, base=1.0) for k, v in bts.items() if k != "股票池等权"}
    pd.DataFrame(rows_ew).T.to_csv(RES / "performance_vs_equalweight.csv")
    cw = bts["股票池市值加权(含股息)"].nav
    rows_cw = {k: perf_stats(v.nav, cw, base=1.0) for k, v in bts.items() if "市值加权" not in k}
    pd.DataFrame(rows_cw).T.to_csv(RES / "performance_vs_capweight_tr.csv")

    # 分段
    seg = {}
    for lab, (a, b) in {"2016-2021": ("2016-01-01", "2021-12-31"), "2022-2026": ("2022-01-01", C.BT_END)}.items():
        for k, v in bts.items():
            nav = v.nav.loc[a:b]; nav = nav / nav.iloc[0]
            bb = bench.loc[a:b]
            seg[(lab, k)] = perf_stats(nav, bb)
        bb = bench.loc[a:b]
        seg[(lab, C.BENCHMARK_NAME)] = perf_stats(bb / bb.iloc[0], bb)
    pd.DataFrame(seg).T.to_csv(RES / "performance_subperiod.csv")

    yt = yearly_table(main.nav, {C.BENCHMARK_NAME: bench, "股票池市值加权(含股息)": cw, "股票池等权": ew}, nav0=1.0)
    yt.to_csv(RES / "yearly.csv")

    # ---- 敏感性:成本、持股数
    sens = {}
    tg = targets_from(scores["等权合成"])
    for lab, mult in {"零成本": 0.0, "基准成本": 1.0, "两倍成本": 2.0}.items():
        sens[("成本", lab)] = perf_stats(run_backtest(m, tg, cost_mult=mult).nav, bench, base=1.0)
    for n in (30, 50, 100):
        sens[("持股数", f"Top{n}")] = perf_stats(run_backtest(m, targets_from(scores["等权合成"], n)).nav, bench, base=1.0)
    for lab, h in {"按最后价格变现(基准)": 0.0, "再折价30%": 0.3, "全额损失": 1.0}.items():
        sens[("退市持仓处理", lab)] = perf_stats(run_backtest(m, tg, delist_haircut=h).nav, bench, base=1.0)
    for b in (50, 100, 150):
        r = run_backtest(m, targets_buffer(scores["等权合成"], buffer_rank=b))
        sens[("换仓缓冲", f"前{b}名不卖")] = perf_stats(r.nav, bench, base=1.0)
        sens[("换仓缓冲", f"前{b}名不卖")]["年均单边换手"] = r.turnover.iloc[1:].sum() / ((r.nav.index[-1] - r.nav.index[0]).days / 365.25)
    pd.DataFrame(sens).T.to_csv(RES / "sensitivity.csv")
    log("sensitivity done")

    # ---- 图
    pl.plot_nav({"多因子策略(等权合成,Top50)": main.nav, "股票池等权": ew, "股票池市值加权(含股息)": cw, C.BENCHMARK_NAME: bench},
                "多因子策略净值 vs 基准(含交易成本)", "nav_main.png", drawdown=drawdown(main.nav))
    rel = main.nav / cw
    pl.plot_nav({"策略/市值加权(含股息)": rel, "策略/股票池等权": main.nav / ew, "策略/中证800价格指数": main.nav / bench},
                "相对净值(超额收益曲线)", "nav_relative.png", logy=False, drawdown=drawdown(rel))
    pl.plot_nav({k: v.nav for k, v in bts.items()} | {C.BENCHMARK_NAME: bench}, "不同合成方式对比", "nav_variants.png")
    pl.plot_nav({k: v.nav for k, v in single.items()} | {C.BENCHMARK_NAME: bench}, "单因子 Top50 多头净值", "nav_single_factor.png")
    pl.plot_cum_ic_all(ic7, FACTOR_LABELS, "ic_cumulative_all.png")
    for k in ALL:
        pl.plot_ic(ic[k], f"{FACTOR_LABELS[k]} 月度 RankIC", f"ic_{k}.png")
        pl.plot_quantiles(qs[k], FACTOR_LABELS[k], f"quantile_{k}.png")
    pl.plot_ic(comp_ic["等权合成"], "等权合成总分 月度 RankIC", "ic_composite.png")
    pl.plot_quantiles(comp_q, "等权合成总分", "quantile_composite.png")
    pl.plot_corr(corr, FACTOR_LABELS, "factor_corr.png")
    pl.plot_yearly(yt, ["策略", C.BENCHMARK_NAME, "股票池市值加权(含股息)", "股票池等权"], "yearly.png")
    pl.plot_series_bars(main.turnover, "每次调仓单边换手率", "turnover.png")
    log("figures done")

    meta = {
        "n_codes": len(m.codes), "n_signal_dates": len(dates), "first_signal": str(dates[0].date()),
        "last_signal": str(dates[-1].date()), "bt_start": str(idx[0].date()), "bt_end": str(idx[-1].date()),
        "avg_pool": float(ustats["最终股票池"].mean()), "unfilled_total": int(main.unfilled.sum()),
        "delisted_holdings_total": int(main.delisted.sum()),
        "runtime_min": (time.time() - t0) / 60,
    }
    json.dump(meta, open(RES / "meta.json", "w"), ensure_ascii=False, indent=2)
    main.nav.to_csv(RES / "nav_main.csv"); bench.to_csv(RES / "nav_bench.csv"); ew.to_csv(RES / "nav_equalweight.csv")
    cw.to_csv(RES / "nav_capweight_tr.csv")
    pd.DataFrame({d: w for d, w in main.holdings.items()}).T.to_csv(RES / "holdings_main.csv")
    summ_n_out = pd.read_csv(RES / "factor_summary_neutral.csv", index_col=0) if proc_n is not None else None
    write_tables_md(summ, comp_summ, corr, perf, perf_single, pd.DataFrame(rows_ew).T, pd.DataFrame(rows_cw).T,
                    pd.DataFrame(seg).T, yt, pd.DataFrame(sens).T, ustats, meta, decay, size_corr, summ_n_out)
    log(f"ALL DONE in {meta['runtime_min']:.1f} min")
    return meta


def _fmt(df: pd.DataFrame, all_pct: bool = False) -> pd.DataFrame:
    """结果表格式化:比例类列显示为百分数,其余数值保留两位小数。"""
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out.index = [" | ".join(map(str, t)) for t in out.index]
    pct_keys = ["收益", "波动", "回撤", "占比", "胜率", "误差", "年化", "换手"]
    for c in out.columns:
        if "起止" in str(c):
            continue
        num = pd.to_numeric(out[c], errors="coerce")
        if num.notna().sum() == 0:
            continue
        is_pct = all_pct or any(k in str(c) for k in pct_keys)
        if is_pct:
            out[c] = num.map(lambda x: "-" if pd.isna(x) else f"{x*100:.1f}%")
        else:
            out[c] = num.map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    return out


def write_tables_md(summ, comp_summ, corr, perf, perf_single, perf_ew, perf_cw, seg, yt, sens, ustats, meta, decay,
                    size_corr=None, summ_n=None):
    parts = ["# 结果表(自动生成)\n", f"```\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n```\n"]
    parts += ["## 股票池规模(均值)\n", ustats.mean().round(1).to_frame("平均").T.to_markdown(), "\n"]
    parts += ["## 单因子检验\n", _fmt(summ).to_markdown(), "\n"]
    parts += ["## IC 衰减(对未来第 k 个月单月收益的平均 RankIC)\n", decay.round(3).to_markdown(), "\n"]
    if size_corr is not None:
        parts += ["## 各因子与 log 市值的平均截面秩相关\n", size_corr.round(2).to_frame().to_markdown(), "\n"]
    if summ_n is not None:
        parts += ["## 行业+市值中性化后的单因子检验\n", _fmt(summ_n).to_markdown(), "\n"]
    parts += ["## 合成因子 IC\n", _fmt(comp_summ).to_markdown(), "\n"]
    parts += ["## 因子相关性\n", corr.round(2).to_markdown(), "\n"]
    parts += ["## 策略绩效(vs 中证800)\n", _fmt(perf).to_markdown(), "\n"]
    parts += ["## 策略绩效(vs 股票池市值加权含股息)\n", _fmt(perf_cw).to_markdown(), "\n"]
    parts += ["## 策略绩效(vs 股票池等权)\n", _fmt(perf_ew).to_markdown(), "\n"]
    parts += ["## 单因子 Top50 多头\n", _fmt(perf_single).to_markdown(), "\n"]
    parts += ["## 分段绩效\n", _fmt(seg).to_markdown(), "\n"]
    parts += ["## 分年\n", _fmt(yt, all_pct=True).to_markdown(), "\n"]
    parts += ["## 敏感性\n", _fmt(sens).to_markdown(), "\n"]
    (RES / "tables.md").write_text("\n".join(parts), encoding="utf-8")
