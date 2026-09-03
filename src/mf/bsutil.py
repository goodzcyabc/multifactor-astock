"""baostock 薄封装:登录、DataFrame 转换、带重试的查询。"""
import contextlib
import io
import socket
import time

# baostock 走裸 socket 且没有超时参数;不设超时时并发连接会无限期卡住。设了超时后卡住会抛异常,由 query() 重试。
socket.setdefaulttimeout(90)

import pandas as pd
import baostock as bs


def login(retries: int = 6):
    """登录带退避重试:baostock 服务端在并发较高时会偶发"网络接收错误"。"""
    last = None
    for i in range(retries):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                lg = bs.login()
            if lg.error_code == "0":
                return
            last = lg.error_msg
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"baostock login failed after {retries} tries: {last}")


def logout():
    with contextlib.redirect_stdout(io.StringIO()):
        bs.logout()


def to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"baostock query error {rs.error_code}: {rs.error_msg}")
    return pd.DataFrame(rows, columns=rs.fields)


def query(fn, *args, retries=4, **kwargs) -> pd.DataFrame:
    last = None
    for i in range(retries):
        try:
            return to_df(fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
            try:
                logout(); login()
            except Exception:  # noqa: BLE001
                pass
    raise RuntimeError(f"query failed after {retries} retries: {last}")
