"""下载全部原始数据(baostock,免费,无需 token)。可重复运行,已下载的股票会跳过。

产出(data/raw/):
  trade_dates.csv      交易日历
  constituents.csv     每个月末的沪深300/中证500 时点成分股
  index_<code>.csv     指数日线
  industry.csv         证监会行业分类(最新快照)
  stock_basic.csv      上市/退市日期
  bars/<code>.csv      个股日线(后复权收盘价、成交额、换手率、涨跌幅、PE/PB、ST、交易状态)
"""
import sys, time, argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import baostock as bs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mf import config as C            # noqa: E402
from mf import bsutil                  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def month_ends(trade_dates: pd.Series) -> list:
    td = pd.to_datetime(trade_dates)
    return td.groupby(td.dt.to_period("M")).max().dt.strftime("%Y-%m-%d").tolist()


def dl_trade_dates():
    f = C.DATA_RAW / "trade_dates.csv"
    if f.exists():
        return pd.read_csv(f)
    df = bsutil.query(bs.query_trade_dates, start_date=C.DATA_START, end_date=C.DATA_END)
    df = df[df.is_trading_day == "1"][["calendar_date"]].rename(columns={"calendar_date": "date"})
    df.to_csv(f, index=False)
    return df


def _cons_worker(dates):
    bsutil.login()
    out = []
    for i, d in enumerate(dates):
        for name, fn in (("hs300", bs.query_hs300_stocks), ("zz500", bs.query_zz500_stocks)):
            df = bsutil.query(fn, date=d)
            df["index"] = name
            df["asof"] = d
            out.append(df)
        if i % 5 == 0:
            log(f"  cons worker: {i+1}/{len(dates)} dates ({d})")
    bsutil.logout()
    return pd.concat(out, ignore_index=True)


def dl_constituents(dates, nproc):
    f = C.DATA_RAW / "constituents.csv"
    if f.exists():
        return pd.read_csv(f)
    chunks = [dates[i::nproc] for i in range(nproc)]
    with Pool(nproc) as p:
        parts = p.map(_cons_worker, chunks)
    df = pd.concat(parts, ignore_index=True).sort_values(["asof", "index", "code"])
    df.to_csv(f, index=False)
    return df


def dl_indexes():
    bsutil.login()
    for code in C.EXTRA_INDEXES:
        f = C.DATA_RAW / f"index_{code}.csv"
        if f.exists():
            continue
        df = bsutil.query(bs.query_history_k_data_plus, code,
                          "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                          start_date=C.DATA_START, end_date=C.DATA_END, frequency="d")
        df.to_csv(f, index=False)
        log(f"index {code}: {len(df)} rows")
    f = C.DATA_RAW / "industry.csv"
    if not f.exists():
        bsutil.query(bs.query_stock_industry).to_csv(f, index=False)
        log("industry saved")
    bsutil.logout()


def _bars_worker(codes):
    """单个进程:逐只下载。任何一只失败只记录并继续,不让整池崩掉;缺的在下一轮补。"""
    try:
        bsutil.login()
    except Exception as e:  # noqa: BLE001
        print("worker login fail:", e, flush=True)
        return pd.DataFrame()
    basics = []
    for i, code in enumerate(codes):
        f = C.DATA_BARS / f"{code}.csv"
        if f.exists():
            continue
        if i % 25 == 0:
            log(f"  bars worker: {i}/{len(codes)} ({code})")
        try:
            df = bsutil.query(bs.query_history_k_data_plus, code, C.BAR_FIELDS,
                              start_date=C.DATA_START, end_date=C.DATA_END,
                              frequency="d", adjustflag="1")   # 1 = 后复权
            df.insert(1, "code", code)
            df.to_csv(f, index=False)
        except Exception as e:  # noqa: BLE001
            print("bars fail", code, e, flush=True)
            continue
        try:
            basics.append(bsutil.query(bs.query_stock_basic, code=code))
        except Exception as e:  # noqa: BLE001
            print("basic fail", code, e, flush=True)
    try:
        bsutil.logout()
    except Exception:  # noqa: BLE001
        pass
    return pd.concat(basics, ignore_index=True) if basics else pd.DataFrame()


def dl_bars(codes, nproc, max_rounds=4):
    f = C.DATA_RAW / "stock_basic.csv"
    for rnd in range(max_rounds):
        todo = [c for c in codes if not (C.DATA_BARS / f"{c}.csv").exists()]
        if not todo:
            break
        n = min(nproc, max(1, len(todo)))
        log(f"bars round {rnd+1}: {len(codes)} codes total, {len(todo)} to download, nproc={n}")
        chunks = [todo[i::n] for i in range(n)]
        t0 = time.time()
        with Pool(n) as p:
            parts = p.map(_bars_worker, chunks)
        parts = [x for x in parts if isinstance(x, pd.DataFrame) and len(x)]
        if parts:
            basics = pd.concat(parts, ignore_index=True)
            if f.exists() and f.stat().st_size > 0:
                basics = pd.concat([pd.read_csv(f, dtype=str), basics.astype(str)], ignore_index=True)
            basics.drop_duplicates("code", keep="last").to_csv(f, index=False)
        log(f"bars round {rnd+1} done in {(time.time()-t0)/60:.1f} min")
    missing = [c for c in codes if not (C.DATA_BARS / f"{c}.csv").exists()]
    log(f"bars finished; still missing {len(missing)}: {missing[:20]}")


def _basic_worker(codes):
    try:
        bsutil.login()
    except Exception as e:  # noqa: BLE001
        print("basic worker login fail:", e, flush=True)
        return pd.DataFrame()
    out = []
    for c in codes:
        try:
            out.append(bsutil.query(bs.query_stock_basic, code=c))
        except Exception as e:  # noqa: BLE001
            print("basic fail", c, e, flush=True)
    try:
        bsutil.logout()
    except Exception:  # noqa: BLE001
        pass
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def dl_basics_missing(codes, nproc):
    """补齐 stock_basic.csv(上市/退市日):有日线但没有基本信息记录的股票。"""
    f = C.DATA_RAW / "stock_basic.csv"
    have = set()
    if f.exists() and f.stat().st_size > 0:
        try:
            have = set(pd.read_csv(f, dtype=str)["code"])
        except Exception:  # noqa: BLE001
            have = set()
    todo = [c for c in codes if c not in have and (C.DATA_BARS / f"{c}.csv").exists()]
    if not todo:
        log("stock_basic complete")
        return
    n = min(nproc, max(1, len(todo)))
    log(f"stock_basic: {len(todo)} codes to fetch, nproc={n}")
    with Pool(n) as p:
        parts = [x for x in p.map(_basic_worker, [todo[i::n] for i in range(n)]) if len(x)]
    if parts:
        new = pd.concat(parts, ignore_index=True).astype(str)
        if have:
            new = pd.concat([pd.read_csv(f, dtype=str), new], ignore_index=True)
        new.drop_duplicates("code", keep="last").to_csv(f, index=False)
    log("stock_basic done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=12, help="个股日线并发进程数")
    ap.add_argument("--nproc-cons", type=int, default=4, help="成分股查询并发进程数")
    a = ap.parse_args()

    bsutil.login()
    td = dl_trade_dates()
    bsutil.logout()
    log(f"trade dates: {len(td)}")
    dates = month_ends(td["date"])
    log(f"month ends: {len(dates)} ({dates[0]} .. {dates[-1]})")

    cons = dl_constituents(dates, a.nproc_cons)
    codes = sorted(cons["code"].unique())
    log(f"constituents: {len(cons)} rows, {len(codes)} unique codes")

    dl_indexes()
    dl_bars(codes, a.nproc)
    dl_basics_missing(codes, min(a.nproc, 6))
    log("ALL DONE")
