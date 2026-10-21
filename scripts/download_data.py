import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from claimsight.config import get_settings


def check_file(path: Path) -> tuple[bool, int, int]:
    if not path.exists():
        return False, 0, 0
    try:
        df = pd.read_csv(path, nrows=10)
        col_count = len(df.columns)
        with open(path, "rb") as f:
            row_count = sum(1 for _ in f) - 1
        return True, max(row_count, 0), col_count
    except Exception:
        return True, 0, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify ClaimSight datasets.")
    parser.add_argument("--check-only", action="store_true", help="Only verify existing files")
    args = parser.parse_args()

    cfg = get_settings()
    raw_cost_dir = cfg.paths.raw_dir / "cost"
    raw_synpuf_dir = cfg.paths.raw_dir / "synpuf"
    raw_fraud_dir = cfg.paths.raw_dir / "fraud"

    raw_cost_dir.mkdir(parents=True, exist_ok=True)
    raw_synpuf_dir.mkdir(parents=True, exist_ok=True)
    raw_fraud_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ClaimSight Dataset Downloader & Verification")
    print("=" * 70)

    # 1. Cost dataset
    cost_file = raw_cost_dir / "insurance.csv"
    print("\n[1] Cost Dataset")
    print("Source URL: https://www.kaggle.com/datasets/mirichoi0218/insurance")
    print(f"Target file: {cost_file}")
    if not cost_file.exists() and not args.check_only:
        kaggle_cred = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_cred.exists():
            print("Found Kaggle credentials. Downloading via Kaggle API...")
            try:
                import kaggle

                kaggle.api.dataset_download_files(
                    "mirichoi0218/insurance", path=str(raw_cost_dir), unzip=True
                )
            except Exception as e:
                print(f"Kaggle download failed: {e}")
        else:
            print("Kaggle credentials (~/.kaggle/kaggle.json) not found.")
            print("Attempting public download fallback...")
            try:
                mirror_url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
                urllib.request.urlretrieve(mirror_url, cost_file)
                print(f"Downloaded insurance.csv successfully to {cost_file}")
            except Exception as e:
                print(f"Direct download failed: {e}")
                print(
                    "Manual instructions: Download insurance.csv from Kaggle and place in data/raw/cost/insurance.csv"
                )

    # 2. Fraud dataset
    print("\n[2] Fraud Dataset")
    print(
        "Source URL: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis"
    )
    print(f"Target folder: {raw_fraud_dir}")
    fraud_labels = list(raw_fraud_dir.glob("Train-*.csv"))
    if not fraud_labels and not args.check_only:
        kaggle_cred = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_cred.exists():
            print("Found Kaggle credentials. Downloading fraud dataset...")
            try:
                import kaggle

                kaggle.api.dataset_download_files(
                    "rohitrox/healthcare-provider-fraud-detection-analysis",
                    path=str(raw_fraud_dir),
                    unzip=True,
                )
            except Exception as e:
                print(f"Kaggle fraud download failed: {e}")
        else:
            print(
                "Manual instructions: Download dataset from https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis"
            )
            print(
                "Unzip into data/raw/fraud/. Files needed: Train_Beneficiarydata*.csv, Train_Inpatientdata*.csv, Train_Outpatientdata*.csv, Train-*.csv"
            )

    # 3. SynPUF dataset
    print("\n[3] SynPUF Medicare Claims Dataset")
    print(
        "Source URL: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DESample01"
    )
    print(
        "Data Dictionary: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/SynPUF_DUG.pdf"
    )
    print(f"Target folder: {raw_synpuf_dir}")
    print(
        "Manual instructions: Download Sample 1 files: 2008, 2009, 2010 Beneficiary Summary CSVs, Inpatient, Outpatient, Carrier 1 and 2 CSVs. Unzip into data/raw/synpuf/."
    )

    # Verification table
    print("\n" + "=" * 70)
    print("Verification Table")
    print("=" * 70)

    rows = []
    # Check cost
    found, r_cnt, c_cnt = check_file(cost_file)
    rows.append(("Cost (insurance.csv)", found, r_cnt, c_cnt))

    # Check fraud files
    for pattern, label in [
        ("Train_Beneficiarydata*.csv", "Fraud Beneficiary"),
        ("Train_Inpatientdata*.csv", "Fraud Inpatient"),
        ("Train_Outpatientdata*.csv", "Fraud Outpatient"),
        ("Train-*.csv", "Fraud Labels"),
    ]:
        matches = list(raw_fraud_dir.glob(pattern))
        if matches:
            f, r, c = check_file(matches[0])
            rows.append((f"{label} ({matches[0].name})", f, r, c))
        else:
            rows.append((label, False, 0, 0))

    # Check SynPUF files
    for year in ["2008", "2009", "2010"]:
        matches = list(raw_synpuf_dir.glob(f"*{year}*beneficiary*.csv")) or list(
            raw_synpuf_dir.glob(f"*Beneficiary_Summary_File_Sample_1_{year}*.csv")
        )
        if matches:
            f, r, c = check_file(matches[0])
            rows.append((f"SynPUF Beneficiary {year}", f, r, c))
        else:
            rows.append((f"SynPUF Beneficiary {year}", False, 0, 0))

    for cat in ["Inpatient", "Outpatient", "Carrier"]:
        matches = list(raw_synpuf_dir.glob(f"*{cat}*.csv"))
        if matches:
            f, r, c = check_file(matches[0])
            rows.append((f"SynPUF {cat}", f, r, c))
        else:
            rows.append((f"SynPUF {cat}", False, 0, 0))

    header = f"{'Dataset File':<40} | {'Found':<6} | {'Rows':<10} | {'Cols':<6}"
    print(header)
    print("-" * len(header))
    all_found = True
    for name, found, r_cnt, c_cnt in rows:
        print(f"{name:<40} | {str(found):<6} | {r_cnt:<10} | {c_cnt:<6}")
        if not found:
            all_found = False

    print("=" * 70)
    if not all_found:
        print("Note: One or more files are missing from data/raw/.")
        if args.check_only:
            sys.exit(1)


if __name__ == "__main__":
    main()
