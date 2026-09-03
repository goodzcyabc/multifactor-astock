#!/usr/bin/env bash
# 一键复现:下载数据 -> 跑流程 -> 生成图表与结果表
set -e
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 scripts/01_download.py --nproc 16
python3 scripts/02_run_pipeline.py
echo "done. see report/report.md, report/figures/, report/results/tables.md"
