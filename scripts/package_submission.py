"""Build the CW1 submission ZIP (code, notebooks, README, requirements, configs, splits, models, metrics, figures).

Usage: python scripts/package_submission.py [--without-tuning-checkpoints]
Excludes the virtual environment, caches, the raw NASA data (fetched by scripts/download_data.py) and the
CW2 report, which is submitted separately as its own PDF.
Add the demonstration video or its permitted link and your technical record before submitting.
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".venv", "__pycache__", ".ipynb_checkpoints", "raw"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--without-tuning-checkpoints", action="store_true",
                    help="omit models/tune_* (regenerable with scripts/tune.py) to reduce size")
    args = ap.parse_args()
    out = ROOT.parent / "cmapss-rul-submission.zip"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
                   and not EXCLUDE_DIRS & set(p.relative_to(ROOT).parts)
                   and not p.name.startswith(("CW2_", "cw2_"))
                   and not (args.without_tuning_checkpoints and p.parent.name == "models" and p.name.startswith("tune_")))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path("cmapss-rul") / p.relative_to(ROOT))
    print(f"Wrote {out} ({len(files)} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
