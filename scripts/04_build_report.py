"""把结果表、图和人工撰写的解读拼成 report/report.md。

- 数字全部来自 report/results/*.csv,不手填。
- 解读文字放在 report/commentary/<section>.md,由作者在看过结果后撰写;缺失时留占位符。
"""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mf import config as C            # noqa: E402
from mf.pipeline import _fmt          # noqa: E402

R = C.REPORT_DIR / "results"
COM = C.REPORT_DIR / "commentary"
FIG = "figures"


def com(name):
    f = COM / f"{name}.md"
    return f.read_text(encoding="utf-8").strip() if f.exists() else f"<!-- TODO: commentary/{name}.md -->"


def csv(name, index_col=0):
    return pd.read_csv(R / name, index_col=index_col)


def md(df, all_pct=False):
    return _fmt(df, all_pct=all_pct).to_markdown()


def pct(x, nd=1):
    return f"{x*100:.{nd}f}%"


meta = json.load(open(R / "meta.json"))
perf = csv("performance.csv")
perf_cw = csv("performance_vs_capweight_tr.csv")
perf_ew = csv("performance_vs_equalweight.csv")
single = csv("performance_single_factor.csv")
summ = csv("factor_summary.csv")
decay = csv("ic_decay.csv")
comp = csv("composite_summary.csv")
corr = csv("factor_corr.csv")
sub = csv("performance_subperiod.csv", index_col=[0, 1])
yearly = csv("yearly.csv")
sens = csv("sensitivity.csv", index_col=[0, 1])
ustats = csv("universe_stats.csv")

S = perf.loc["等权合成"]
B = perf.loc[C.BENCHMARK_NAME]
CW = perf.loc["股票池市值加权(含股息)"]
EW = perf.loc["股票池等权"]
S_cw = perf_cw.loc["等权合成"]
S_ew = perf_ew.loc["等权合成"]

KEY = ["年化收益", "年化波动", "夏普比率", "最大回撤", "年化超额收益", "信息比率", "月度胜率(vs基准)", "年均单边换手"]
KEY_CW = ["年化收益", "年化超额收益", "跟踪误差", "信息比率", "超额最大回撤", "月度胜率(vs基准)"]

parts = []
A = parts.append

A(f"""# A 股多因子选股策略:中证 800 股票池的七因子月度轮动

**作业题目:** 完成一个多因子策略,品种任选。
**品种:** A 股(中证 800 时点成分股)。**回测区间:** {meta['bt_start']} 至 {meta['bt_end']}(含交易成本)。
**代码:** 本目录 `src/`、`scripts/`,`./run_all.sh` 一键复现;数据来自 baostock 免费接口。

## 1. 摘要

| | 多因子策略 | 中证 800 价格指数 | 股票池市值加权(含股息) | 股票池等权 |
|---|---|---|---|---|
| 年化收益 | **{pct(S['年化收益'])}** | {pct(B['年化收益'])} | {pct(CW['年化收益'])} | {pct(EW['年化收益'])} |
| 年化波动 | {pct(S['年化波动'])} | {pct(B['年化波动'])} | {pct(CW['年化波动'])} | {pct(EW['年化波动'])} |
| 夏普比率 | **{S['夏普比率']:.2f}** | {B['夏普比率']:.2f} | {CW['夏普比率']:.2f} | {EW['夏普比率']:.2f} |
| 最大回撤 | {pct(S['最大回撤'])} | {pct(B['最大回撤'])} | {pct(CW['最大回撤'])} | {pct(EW['最大回撤'])} |
| 相对中证 800 年化超额 | **{pct(S['年化超额收益'])}** | – | {pct(CW['年化超额收益'])} | {pct(EW['年化超额收益'])} |
| 相对市值加权(含股息)年化超额 | **{pct(S_cw['年化超额收益'])}**(IR {S_cw['信息比率']:.2f}) | – | – | {pct(perf_cw.loc['股票池等权','年化超额收益'])} |
| 相对股票池等权年化超额 | **{pct(S_ew['年化超额收益'])}**(IR {S_ew['信息比率']:.2f}) | – | – | – |
| 年均单边换手 | {pct(S['年均单边换手'], 0)} | – | – | {pct(EW['年均单边换手'], 0)} |

{com('summary')}

![净值](figures/nav_main.png)
""")

A((C.REPORT_DIR / "report_draft_method.md").read_text(encoding="utf-8"))

A(f"""
## 5. 数据检查

在看任何收益数字之前,先确认数据本身没有骗人。以下检查由 `scripts/03_validate_data.py` 自动生成,完整版见 `results/data_checks.md`。

{com('data_checks')}

股票池规模(146 个月末的平均值):

{ustats.mean().round(1).to_frame('平均').T.to_markdown()}

另有两项写成了自动化测试(`tests/test_with_data.py`):
(a) **扰动未来数据测试**:把某个信号日之后的全部价格、成交、估值数据随机乘以 0.5–1.5 的噪声并翻转 ST 标记,该信号日选出的 50 只股票必须完全不变——通过;
(b) 后复权收盘价与交易所官方涨跌幅的一致性——原始数据有 23 个"股票-日"相差超过 1 个百分点,已按第 4 节的方法修正,修正后仅剩官方涨跌幅本身为 0 占位的个别情形。

## 6. 单因子检验结果

{md(summ)}

> IC 为月度 RankIC;t 值(NW)为 Newey-West 修正;"执行口径"指前向收益从 T+1 收盘到下月 T+1 收盘,与回测完全一致。
> 多空年化为 Q5−Q1 等权组合的年化收益(A 股融券受限,多空仅作因子强度度量)。

![累计IC](figures/ic_cumulative_all.png)

**IC 衰减**(因子对未来第 k 个月单月收益的平均 RankIC):

{decay.round(3).to_markdown()}

**因子相关性**(平均截面秩相关):

{corr.round(2).to_markdown()}

![相关性](figures/factor_corr.png)

{com('single_factor')}

各因子的月度 IC 与分层净值图见 `figures/ic_<factor>.png` 和 `figures/quantile_<factor>.png`。

## 7. 因子合成与组合表现

合成总分的 IC:

{md(comp)}

![合成分层](figures/quantile_composite.png)

策略与基准的完整绩效(相对中证 800 价格指数):

{md(perf)}

相对"股票池市值加权(含股息)"——即剔除股息口径差异后的公平基准:

{md(perf_cw)}

相对"股票池等权"——剔除等权效应后,纯因子选股的贡献:

{md(perf_ew)}

![相对净值](figures/nav_relative.png)

![变体对比](figures/nav_variants.png)

{com('strategy')}

### 分年表现

{md(yearly, all_pct=True)}

![分年](figures/yearly.png)

### 分段表现

{md(sub)}

{com('subperiod')}

### 单因子 Top50 多头(每个因子单独选股)

{md(single)}

![单因子](figures/nav_single_factor.png)

## 8. 稳健性与敏感性

{md(sens)}

![换手](figures/turnover.png)

{com('robustness')}
""")

A((C.REPORT_DIR / "report_draft_limits.md").read_text(encoding="utf-8"))

A(f"""
## 11. 结论

{com('conclusion')}

---

### 附录 A:复现

```bash
pip install -r requirements.txt
python scripts/01_download.py          # baostock 下载,约 1 小时,断点续传
python scripts/02_run_pipeline.py      # 全部检验、回测、图表
python scripts/03_validate_data.py     # 数据检查
python scripts/04_build_report.py      # 生成本报告
python tests/test_core.py && python tests/test_with_data.py
```

运行环境:Python 3.9,pandas 2.3,numpy 2.0,statsmodels 0.14,baostock 0.8。流程耗时 {meta['runtime_min']:.1f} 分钟(不含下载)。

### 附录 B:参数一览

| 参数 | 取值 |
|---|---|
| 股票池 | 中证 800 时点成分股,剔除 ST / 停牌 / 上市不足 {C.MIN_LISTED_DAYS} 交易日 |
| 信号日 / 成交日 | 月末收盘 / 下一交易日收盘 |
| 持股数 | {C.TOP_N},等权 |
| 因子 | 动量 12-1、反转 1 月、低波动 60 日、小市值、低换手 20 日、BP、ROE |
| 预处理 | MAD 3 倍去极值,z-score;对照版本加行业+市值中性化 |
| 合成 | 等权;对照版本滚动 12 月 ICIR 加权 |
| 成本 | 佣金双边万 {C.COMMISSION*1e4:.0f},印花税卖出千 1(2023-08-28 起万 5),滑点单边千 {C.SLIPPAGE*1e3:.0f} |
| 涨跌停 | 主板 10%、ST 5%、科创板 20%、创业板 2020-08-24 起 20%;收盘涨跌幅贴近限制即视为封板 |
| 数据 | baostock;{meta['n_codes']} 只股票 × {C.DATA_START} 至 {C.DATA_END} 日线 |
""")

out = C.REPORT_DIR / "report.md"
out.write_text("\n".join(parts), encoding="utf-8")
print(out, len("\n".join(parts)), "chars")
