"""跑完整流程(需先运行 01_download.py)。用法: python scripts/02_run_pipeline.py [--force] [--no-neutral]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mf.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="重建 parquet 缓存")
    ap.add_argument("--no-neutral", action="store_true", help="跳过行业市值中性化变体(更快)")
    a = ap.parse_args()
    run(force=a.force, neutral_variant=not a.no_neutral)
