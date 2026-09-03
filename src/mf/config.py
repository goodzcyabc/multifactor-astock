"""全局配置:路径、时间区间、股票池、回测参数。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_BARS = DATA_RAW / "bars"
DATA_PROC = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
for _d in (DATA_BARS, DATA_PROC, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 数据下载区间:回测从 2016 年初开始,再往前多留 18 个月给 12 个月动量等长窗口因子预热
DATA_START = "2014-07-01"
DATA_END = "2026-08-31"

# 回测区间
BT_START = "2016-01-01"
BT_END = "2026-08-31"

# 股票池:中证 800 = 沪深 300 + 中证 500,按月取"当时"的成分股(时点成分股,避免幸存者偏差)
UNIVERSE_INDEXES = {"hs300": "sh.000300", "zz500": "sh.000905"}
BENCHMARK_CODE = "sh.000906"          # 中证 800 全收益不可得,用价格指数近似
BENCHMARK_NAME = "中证800"
EXTRA_INDEXES = ["sh.000300", "sh.000905", "sh.000906"]

# baostock 日线字段。注意:baostock 的耗时与返回字节数成正比(实测 4 个字段 3.5 秒,16 个字段 31.6 秒),
# 因此只取策略真正用到的字段:后复权收盘价、成交额、换手率、交易状态、涨跌幅、ST、PE、PB。代码由文件名补回。
BAR_FIELDS = "date,close,amount,turn,tradestatus,pctChg,isST,peTTM,pbMRQ"

# 可交易性过滤
MIN_LISTED_DAYS = 120       # 上市不满 120 个交易日的新股不进入
LIMIT_MOVE_PCT = 9.5        # 涨跌幅绝对值超过该值视为触及涨跌停,不能在当天成交

# 组合与成本
REBALANCE_FREQ = "M"        # 月度调仓
TOP_N = 50                  # 做多因子总分最高的 50 只
BUFFER_RANK = 100           # 换仓缓冲变体:已持有股票排名仍在前 100 名内就不卖
N_QUANTILES = 5             # 分层回测分 5 组
COMMISSION = 0.0003         # 佣金双边各万 3
SLIPPAGE = 0.001            # 滑点单边千 1
# 印花税(仅卖出):2023-08-28 之前千 1,之后减半为万 5。回测按成交日期取对应税率。
STAMP_TAX_SCHEDULE = [("1900-01-01", 0.0010), ("2023-08-28", 0.0005)]


def stamp_tax_rate(date) -> float:
    import pandas as pd
    d = pd.Timestamp(date)
    rate = STAMP_TAX_SCHEDULE[0][1]
    for start, r in STAMP_TAX_SCHEDULE:
        if d >= pd.Timestamp(start):
            rate = r
    return rate
