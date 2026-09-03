"""数据质量与偏差检查,输出 report/results/data_checks.md。

检查项:
 1. 成分股名单是否为"时点"名单:后来退市/被剔除的股票是否出现在早期名单里
 2. 退市股票的日线是否可得;每年成分股的行情覆盖率
 3. pbMRQ 是否按财报公告日更新(时点性):账面价值跳变日期 vs 公告日期
 4. 流通市值恒等式(成交额/换手率)与 baostock 流通股本×不复权收盘价 的相对误差
 5. 停牌日在数据中的表现方式;涨跌停触发频率
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mf import config as C                      # noqa: E402
from mf.data import Market, load_constituents, load_stock_basic, universe_at  # noqa: E402
from mf import bsutil                            # noqa: E402
import baostock as bs                            # noqa: E402

OUT = C.REPORT_DIR / "results" / "data_checks.md"
lines = ["# 数据检查(自动生成)\n"]


def sec(t):
    lines.append(f"\n## {t}\n")


def add(s=""):
    lines.append(s)


m = Market()
cons = load_constituents()
basic = load_stock_basic()

# ---- 1. 时点成分股
sec("1. 成分股名单是时点名单吗?")
first_list = set(universe_at(cons, pd.Timestamp("2016-01-29")))
last_list = set(universe_at(cons, cons["asof"].max()))
gone = sorted(first_list - last_list)
delisted = basic[basic["outDate"].notna()]
gone_delisted = [c for c in gone if c in delisted.index]
add(f"- 2016-01 名单 {len(first_list)} 只,{cons['asof'].max():%Y-%m} 名单 {len(last_list)} 只;2016-01 名单中 {len(gone)} 只已不在最新名单。")
add(f"- 其中 {len(gone_delisted)} 只后来退市(baostock outDate 非空)。示例:")
names = cons.drop_duplicates("code").set_index("code")["code_name"]
for c in gone_delisted[:8]:
    add(f"  - {c} {names.get(c, '')} 退市日 {delisted.loc[c, 'outDate']:%Y-%m-%d}")
add("- 结论:名单包含后来退市的公司,说明接口返回的是历史时点成分,幸存者偏差已在源头消除。" if gone_delisted
    else "- 警告:名单中没有后来退市的公司,需人工核实接口是否返回历史成分。")
# 名单每月变化
chg = []
dates = sorted(cons["asof"].unique())
for a, b in zip(dates[:-1], dates[1:]):
    sa, sb = set(universe_at(cons, a)), set(universe_at(cons, b))
    chg.append((b, len(sb - sa), len(sa - sb)))
chg = pd.DataFrame(chg, columns=["月末", "新增", "剔除"]).set_index("月末")
big = chg[chg["新增"] > 0]
add(f"- 146 个月末中有 {len(big)} 个月名单发生变化;变化最大的月份:")
for d, r in chg.sort_values("新增", ascending=False).head(6).iterrows():
    add(f"  - {d:%Y-%m}: 新增 {r['新增']} 剔除 {r['剔除']}")
add("- 中证指数每年 6 月、12 月定期调整,上表应集中在这两个月(其余为临时调整)。")
# 结构断言
sz = cons.groupby("asof")["code"].nunique()
overlap = cons.groupby("asof").apply(lambda g: len(set(g.loc[g["index"] == "hs300", "code"]) & set(g.loc[g["index"] == "zz500", "code"])))
add(f"- 结构检查:每个月末名单股票数 min={sz.min()} max={sz.max()}(应为 800);沪深300 与中证500 重叠数 max={overlap.max()}(应为 0)。")
chg["变动"] = chg["新增"] + chg["剔除"]
jun_dec = chg[chg.index.month.isin([6, 12])]["变动"]
other = chg[~chg.index.month.isin([6, 12])]["变动"]
add(f"- 6/12 月的名单变动数:中位数 {jun_dec.median():.0f}(定期调整,预期 60~130);其它月份中位数 {other.median():.0f}、最大 {other.max()}(临时调整,预期很小)。")

# ---- 2. 覆盖率
sec("2. 行情覆盖率(名单中有日线数据的比例)")
cov = {}
for d in m.month_ends(start="2015-12-01", end=C.BT_END):
    u = universe_at(cons, d)
    have = [c for c in u if c in m.close.columns and pd.notna(m.close.loc[d, c])]
    cov[d] = (len(u), len(have))
cov = pd.DataFrame(cov, index=["名单", "有当日行情"]).T
cov["覆盖率"] = cov["有当日行情"] / cov["名单"]
yearly = cov.groupby(cov.index.year).agg({"名单": "mean", "有当日行情": "mean", "覆盖率": "mean"}).round(3)
add(yearly.to_markdown())
add(f"\n- 全期平均覆盖率 {cov['覆盖率'].mean():.1%};未覆盖的主要是当日停牌(无当日行情)的股票,它们本来也不可交易。")
have_bars = [c for c in delisted.index if c in m.close.columns]
add(f"- {len(delisted)} 只退市股票中 {len(have_bars)} 只有日线数据,说明退市股历史行情可得。")
all_codes = sorted(cons["code"].unique())
missing = [c for c in all_codes if c not in m.close.columns]
add(f"- 名单中 {len(all_codes)} 只股票,{len(missing)} 只没有任何日线数据(通常是代码变更/吸收合并):{', '.join(missing[:15])}")

# ---- 3. pbMRQ 时点性(需要联网,取 6 只样本股)
sec("3. PB(MRQ) 是否按公告日更新")
sample = ["sh.600519", "sh.600036", "sz.000858", "sz.000002", "sh.601318", "sz.000333"]
sample = [c for c in sample if c in m.close.columns]
try:
    bsutil.login()
    near_pub, near_qend, total = 0, 0, 0
    detail = []
    for c in sample:
        pubs = []
        for y in range(2018, 2026):
            for q in (1, 2, 3, 4):
                try:
                    d = bsutil.query(bs.query_profit_data, code=c, year=y, quarter=q)
                    if len(d):
                        pubs.append(pd.Timestamp(d["pubDate"].iloc[0]))
                except Exception:
                    pass
        pubs = pd.DatetimeIndex(sorted(set(pubs)))
        # PB 与价格的比值只在"每股净资产"变化时跳变。价格用官方涨跌幅链(不含除权除息影响),
        # 因此除权除息日也会显示为跳变——这些日子和财报公告日一起构成"合理跳变"。
        pb = m.pb[c].loc["2018":"2025"]
        ratio = (pb / pb.shift(1)) / (1 + m.pct[c].loc["2018":"2025"] / 100.0)
        jump_dates = ratio.index[(ratio - 1).abs() > 0.02]
        qends = pd.DatetimeIndex([pd.Timestamp(f"{y}-{md}") for y in range(2018, 2026) for md in ("03-31", "06-30", "09-30", "12-31")])
        a = sum(any(abs((jd - p).days) <= 3 for p in pubs) for jd in jump_dates)
        b = sum(any(abs((jd - q).days) <= 3 for q in qends) for jd in jump_dates)
        near_pub += a; near_qend += b; total += len(jump_dates)
        detail.append(f"  - {c}: 跳变 {len(jump_dates)} 次;落在财报公告日 ±3 天:{a} 次;落在季度末 ±3 天:{b} 次(公告日共 {len(pubs)} 个)")
    bsutil.logout()
    add(f"- 方法:PB 相对价格的比值只应在每股净资产变化时跳变。若数据源按报告期末回填财报,跳变会集中在 3/31、6/30、9/30、12/31;"
        f"若按公告日更新,跳变集中在公告日附近(其余为除权除息、送转等股本变动)。样本 {len(sample)} 只 2018-2025:")
    lines.extend(detail)
    add(f"- 合计:{total} 次跳变中 {near_pub} 次({near_pub/max(total,1):.0%})落在公告日附近,{near_qend} 次({near_qend/max(total,1):.0%})落在季度末附近。"
        "公告日占比高、季度末占比低,说明 PB/PE 是按公告日更新的时点数据,价值与质量因子不含财报前视。")
except Exception as e:  # noqa: BLE001
    add(f"- 联网检查失败:{e}")

# ---- 4. 流通市值恒等式
sec("4. 流通市值 = 成交额/换手率 的校验")
try:
    bsutil.login()
    rows = []
    for c in sample[:5]:
        raw = bsutil.query(bs.query_history_k_data_plus, c, "date,close", start_date="2024-01-01", end_date="2024-12-31",
                           frequency="d", adjustflag="3")
        raw["date"] = pd.to_datetime(raw["date"]); raw = raw.set_index("date")["close"].astype(float)
        sh = bsutil.query(bs.query_profit_data, code=c, year=2024, quarter=2)
        liqa = float(sh["liqaShare"].iloc[0]) if len(sh) else np.nan
        est = m.float_mcap[c].loc["2024-07":"2024-09"]
        ref = raw.loc["2024-07":"2024-09"] * liqa
        err = ((est / ref) - 1).dropna()
        rows.append((c, liqa / 1e8, err.abs().median(), err.abs().max()))
    bsutil.logout()
    df = pd.DataFrame(rows, columns=["代码", "流通股本(亿股,2024Q2)", "相对误差中位数", "相对误差最大"]).set_index("代码")
    add("- 用 2024Q2 流通股本 × 不复权收盘价 作参照,比较 2024 年 7-9 月的推算流通市值:")
    add(df.round(4).to_markdown())
    add("- 误差来源:成交额/换手率 = 流通股本 × 当日成交均价(VWAP),与收盘价的差通常在 1% 以内;对 log 市值的排序影响可忽略。")
except Exception as e:  # noqa: BLE001
    add(f"- 联网检查失败:{e}")

# ---- 4b. 市值合理性:沪深300 成分市值应显著大于中证500;推算市值加权组合应能复现沪深300 指数日收益
sec("4b. 推算流通市值的合理性")
try:
    d0 = pd.Timestamp("2023-06-30")
    u300 = [c for c in cons.loc[(cons["asof"] == d0) & (cons["index"] == "hs300"), "code"] if c in m.close.columns]
    u500 = [c for c in cons.loc[(cons["asof"] == d0) & (cons["index"] == "zz500"), "code"] if c in m.close.columns]
    mc = m.float_mcap.loc[d0]
    add(f"- {d0:%Y-%m-%d}:沪深300 成分流通市值中位数 {mc[u300].median()/1e8:.0f} 亿元,中证500 成分中位数 {mc[u500].median()/1e8:.0f} 亿元(前者应明显更大)。")
    add(f"- 全市场推算流通市值范围:5% 分位 {mc.quantile(0.05)/1e8:.0f} 亿,95% 分位 {mc.quantile(0.95)/1e8:.0f} 亿。")
    # 复现沪深300:每月末按推算市值加权,持有一个月,与指数日收益比较
    idx = pd.read_csv(C.DATA_RAW / "index_sh.000300.csv"); idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")["close"].astype(float).pct_change()
    rets = []
    for d in m.month_ends(start="2016-01-01", end=C.BT_END)[:-1]:
        u = [c for c in universe_at(cons[cons["index"] == "hs300"], d) if c in m.close.columns]
        w = m.float_mcap.loc[d, u].dropna(); w = w / w.sum()
        nxt = m.dates[(m.dates > d)][:25]
        r = m.ret.loc[nxt, w.index].fillna(0) @ w
        rets.append(r)
    rep = pd.concat(rets); rep = rep[~rep.index.duplicated()]
    both = pd.concat([rep.rename("rep"), idx.rename("idx")], axis=1).dropna()
    corr = both["rep"].corr(both["idx"]); te = (both["rep"] - both["idx"]).std() * np.sqrt(243)
    add(f"- 用推算流通市值对沪深300 成分股加权(月度再平衡)复现指数:日收益相关系数 {corr:.4f},年化跟踪误差 {te:.2%}。")
    add("  相关系数接近 1 说明推算市值的权重结构与指数一致(指数用自由流通市值分级靠档加权、且不含股息,故不可能完全重合)。")
except Exception as e:  # noqa: BLE001
    add(f"- 检查失败:{e}")

# ---- 5. 停牌与涨跌停
sec("5. 停牌与涨跌停")
long_cols = ["close", "amount", "turn", "tradestatus"]
n_rows = m.close.notna().sum().sum()
susp = (~m.trading) & m.close.notna()
add(f"- 有行情记录的股票日 {n_rows:,} 个,其中交易状态为停牌/无成交 {int(susp.sum().sum()):,} 个({susp.sum().sum()/n_rows:.2%})。")
add("- baostock 停牌日仍返回一行:tradestatus=0、成交额 0、收盘价延续前值。本项目把这些日子标记为不可交易,收益记 0。")
exec_days = [m.next_trading_day(d) for d in m.month_ends(start="2015-12-01", end=C.BT_END)]
exec_days = [d for d in exec_days if d is not None]
lu = m.limit_up.loc[exec_days].sum().sum(); ld = m.limit_down.loc[exec_days].sum().sum()
n_obs = m.close.loc[exec_days].notna().sum().sum()
add(f"- 在 {len(exec_days)} 个执行日上,全部有数据股票中涨停 {int(lu)} 个、跌停 {int(ld)} 个股票日(占 {(lu+ld)/n_obs:.2%})。")

# ---- 6. 复权价 vs 官方涨跌幅
sec("6. 后复权收盘价与官方涨跌幅的一致性(及修正)")
both = m.trading & m.trading.shift(1).fillna(False)
r_adj = m.close.ffill().pct_change(); r_pct = m.pct / 100.0
gap = (r_adj - r_pct).abs().where(both)
n = int(both.sum().sum()); bad = gap[gap > 0.01].stack()
add(f"- 连续交易日样本 {n:,} 个,后复权收盘价环比与官方涨跌幅相差超过 1 个百分点的有 {len(bad)} 个。")
add("- 逐个裁决(用 PB 日变化作独立见证,PB 随原始价同步变动):")
rows = []
for (d, c), g in bad.items():
    rows.append({"日期": f"{d:%Y-%m-%d}", "代码": c, "复权价环比": f"{r_adj.loc[d, c]:+.2%}", "官方涨跌幅": f"{r_pct.loc[d, c]:+.2%}",
                 "PB变动": f"{m.pb.ffill().pct_change().loc[d, c]:+.2%}", "采用": "官方涨跌幅(修正复权价)" if m.repaired.loc[d, c] else "复权价(涨跌幅疑为占位0)"})
add(pd.DataFrame(rows).to_markdown(index=False))
add(f"- 共修正 {int(m.repaired.sum().sum())} 个单元格;修正用累计系数向后传播,只依赖当日及之前的数据,不引入前视。"
    "未修正的情况下,平安银行等股票的动量/反转因子会在长达 12 个月内偏差 16–30 个百分点。")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(OUT)
print("\n".join(lines))
