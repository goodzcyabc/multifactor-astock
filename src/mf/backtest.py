"""日频权重回测。

时间线:信号日 T(月末)收盘后算分 -> T+1 收盘价成交 -> 持有到下个月的 T'+1 收盘。
执行约束:T+1 停牌或涨停的股票买不进(权重分给其它可买目标);持仓中 T+1 停牌或跌停的股票卖不出(冻结,保持仓位)。
成本:买入 佣金+滑点;卖出 佣金+滑点+印花税(印花税按成交日期分段:2023-08-28 前千 1,之后万 5)。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .data import Market


@dataclass
class BacktestResult:
    nav: pd.Series
    daily_ret: pd.Series
    turnover: pd.Series          # 每个执行日的单边换手(买入额+卖出额)/2
    cost: pd.Series              # 每个执行日的成本(占净值比例)
    n_hold: pd.Series
    holdings: dict = field(default_factory=dict)   # 执行日 -> 权重
    unfilled: pd.Series = None   # 每个执行日因涨停/停牌未买到的目标数
    delisted: pd.Series = None   # 每个执行日因数据序列终止(退市/换代码)而按最后价格变现的持仓数


def run_backtest(m: Market, targets: dict, end=None, commission=C.COMMISSION, stamp_tax=None,
                 slippage=C.SLIPPAGE, cost_mult: float = 1.0, delist_haircut: float = 0.0) -> BacktestResult:
    """targets: {信号日 -> pd.Series(code -> 目标权重, 和为 1)}
    stamp_tax: None 表示按 config.STAMP_TAX_SCHEDULE 随成交日期取值;给定数值则固定。
    cost_mult: 所有成本的统一倍数(0 = 无成本,2 = 两倍成本),用于敏感性分析。
    delist_haircut: 持仓股数据序列终止(退市/换代码)时,在最后价格基础上再折价的比例(0 = 按最后价格变现,
                    1 = 全额损失),用于退市处理的敏感性分析。"""
    end = pd.Timestamp(end or C.BT_END)
    exec_map = {}
    for t in sorted(targets):
        e = m.next_trading_day(pd.Timestamp(t))
        if e is not None and e <= end:
            exec_map[e] = t
    if not exec_map:
        raise ValueError("no executable rebalance dates")
    first = min(exec_map)
    dates = m.dates[(m.dates >= first) & (m.dates <= end)]

    w = pd.Series(dtype=float)
    V = 1.0
    nav, dret, turn, cost_s, nh, unfilled, holdings, delist = {}, {}, {}, {}, {}, {}, {}, {}
    # 净值序列从首个执行日开始,首日值 = 1 - 建仓成本(初始资金为 1.0;绩效统计以 1.0 为基数)

    for d in dates:
        port_r = 0.0
        if len(w):
            r = m.ret.loc[d, w.index].fillna(0.0)
            port_r = float((w * r).sum())
            V *= 1.0 + port_r
            w = w * (1.0 + r) / (1.0 + port_r)

        if d in exec_map:
            tgt = targets[exec_map[d]].copy()
            tgt = tgt[tgt > 0]
            held = w.index
            # 不能卖:停牌或跌停(且仍有数据,若数据已结束视为已退市按最后价格变现)
            if len(held):
                untradable = ~m.trading.loc[d, held].fillna(False).astype(bool)
                lim_dn = m.limit_down.loc[d, held].fillna(False).astype(bool)
                gone = pd.Series(m.last_date.reindex(held).values < d, index=held)
                frozen_mask = (untradable | lim_dn) & ~gone
                frozen = w[frozen_mask.values]
                delist[d] = int(gone.sum())
                if gone.any():
                    # 序列终止的持仓:按最后价格(再打折 delist_haircut)变现为现金,不算交易成本
                    w_gone = float(w[gone.values].sum())
                    V *= 1.0 - delist_haircut * w_gone
                    w = w[~gone.values] / (1.0 - delist_haircut * w_gone) if (1.0 - delist_haircut * w_gone) > 0 else w[~gone.values]
            else:
                frozen = pd.Series(dtype=float)
                delist[d] = 0
            # 不能买:停牌或涨停。已持有且仍在目标中的涨停股不卖出(涨停能卖但没理由卖),按当前权重继续持有
            tradable = m.trading.loc[d, tgt.index].fillna(False).astype(bool)
            lim_up = m.limit_up.loc[d, tgt.index].fillna(False).astype(bool)
            keep = tgt.index[(lim_up.values) & tgt.index.isin(w.index) & ~tgt.index.isin(frozen.index)]
            if len(keep):
                frozen = pd.concat([frozen, w[keep]])
            buyable = tgt[(tradable & ~lim_up).values]
            n_unfilled = len(tgt) - len(buyable) - len(keep)
            buyable = buyable.drop(index=buyable.index.intersection(frozen.index))
            avail = max(1.0 - float(frozen.sum()), 0.0)
            if buyable.sum() > 0:
                buyable = buyable / buyable.sum() * avail
            new_w = pd.concat([frozen, buyable]).groupby(level=0).sum()

            allc = w.index.union(new_w.index)
            dw = new_w.reindex(allc).fillna(0.0) - w.reindex(allc).fillna(0.0)
            buys, sells = float(dw[dw > 0].sum()), float(-dw[dw < 0].sum())
            st = C.stamp_tax_rate(d) if stamp_tax is None else stamp_tax
            cost = cost_mult * (buys * (commission + slippage) + sells * (commission + slippage + st))
            V *= 1.0 - cost
            w = new_w[new_w > 0]
            turn[d] = (buys + sells) / 2.0
            cost_s[d] = cost
            nh[d] = len(w)
            unfilled[d] = n_unfilled
            holdings[d] = w.copy()

        nav[d] = V
        dret[d] = port_r

    nav = pd.Series(nav, name="nav")
    dr = nav.pct_change().fillna(0.0)   # 含成本的日收益
    return BacktestResult(nav=nav, daily_ret=dr, turnover=pd.Series(turn), cost=pd.Series(cost_s),
                          n_hold=pd.Series(nh), holdings=holdings, unfilled=pd.Series(unfilled),
                          delisted=pd.Series(delist))
