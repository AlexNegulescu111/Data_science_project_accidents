"""Run the whole project in order.

    python run_pipeline.py                 # clean -> process -> analysis -> models
    python run_pipeline.py --skip-models   # without the (slower) models
    python run_pipeline.py --scrape        # scrape the site first (needs Chrome)
    python run_pipeline.py --show          # also open every figure on screen
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ETL = ["scripts/clean_data.py", "scripts/process_data.py"]
ANALYSIS = [
    "analysis/eda.py",
    "analysis/risk_analysis.py",
    "analysis/vehicle_analysis.py",
]
MODELS = ["scripts/train_fatality_classifier.py", "scripts/train_resumption_regressor.py"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scrape", action="store_true", help="run the scraper first")
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--show", action="store_true", help="show figures on screen (they are always saved)")
    args = p.parse_args()

    steps = (["scripts/scrape_data.py"] if args.scrape else []) + ETL + ANALYSIS
    if not args.skip_models:
        steps += MODELS

    env = dict(os.environ, SHOW_PLOTS="1" if args.show else "0")
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT, env=env)
        if result.returncode != 0:
            sys.exit(f"Step {step} failed (exit code {result.returncode})")
        print(f"--- {step} done in {time.time() - start:.1f}s")
    print("\nPipeline finished. Figures: visuals/, metrics: reports/")


if __name__ == "__main__":
    main()
