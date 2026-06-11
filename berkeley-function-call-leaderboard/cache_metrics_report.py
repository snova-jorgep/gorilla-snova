import argparse
import csv
import json
import os
import uuid
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv

from utils import upload_to_s3

base_dir = Path(__file__).resolve().parent
load_dotenv(base_dir / ".env")
load_dotenv()

S3_PREFIX = "fc-so-testing-suite/bfcl_cache_metrics"
CSV_HEADERS = [
    "run_id",
    "date",
    "provider",
    "model",
    "test_category",
    "total_input_tokens",
    "total_output_tokens",
    "total_cached_tokens",
    "cache_hit_rate_pct",
    "n_entries",
]


def _flatten_token_count(value) -> int:
    """Flatten scalar, list, or nested list of token counts to a single int."""
    if isinstance(value, list):
        return sum(_flatten_token_count(v) for v in value)
    return int(value or 0)


def generate_cache_metrics_report(
    result_base_dir: Path,
    date: str,
    run_id: str = None,
) -> Path:
    if run_id is None:
        run_id = str(uuid.uuid4())

    rows = []

    # Walk result/{provider}/{date}/{model}/BFCL_v3_{category}_result.json
    for provider_dir in sorted(result_base_dir.iterdir()):
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name

        date_dir = provider_dir / date
        if not date_dir.exists():
            continue

        for model_dir in sorted(date_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name

            # Accumulate per category
            category_stats: dict[str, dict] = {}

            for result_file in sorted(model_dir.glob("*_result.json")):
                # Extract category from filename: BFCL_v3_{category}_result.json
                stem = result_file.stem  # e.g. BFCL_v3_multi_turn_base_result
                category = stem.replace("BFCL_v3_", "").replace("_result", "")

                stats = category_stats.setdefault(
                    category,
                    {"input": 0, "output": 0, "cached": 0, "n": 0},
                )

                with open(result_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        stats["input"] += _flatten_token_count(
                            entry.get("input_token_count", 0)
                        )
                        stats["output"] += _flatten_token_count(
                            entry.get("output_token_count", 0)
                        )
                        stats["cached"] += _flatten_token_count(
                            entry.get("cached_token_count", 0)
                        )
                        stats["n"] += 1

            for category, stats in category_stats.items():
                total_input = stats["input"]
                total_cached = stats["cached"]
                hit_rate = (
                    round(total_cached / total_input * 100, 2) if total_input > 0 else 0.0
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "date": date,
                        "provider": provider,
                        "model": model,
                        "test_category": category,
                        "total_input_tokens": total_input,
                        "total_output_tokens": stats["output"],
                        "total_cached_tokens": total_cached,
                        "cache_hit_rate_pct": hit_rate,
                        "n_entries": stats["n"],
                    }
                )

    if not rows:
        print("No result files found — cache metrics report skipped.")
        return None

    output_dir = base_dir / "results" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cache_metrics.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cache metrics written to: {output_path}")

    s3_key = f"{S3_PREFIX}/{date}/cache_metrics.csv"
    upload_to_s3(output_path, s3_key)
    print(f"Uploaded to s3://.../{s3_key}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BFCL cache metrics report")
    parser.add_argument(
        "--run-date",
        type=str,
        required=True,
        help="Run date folder name under result/{provider}/ (e.g. 2025-06-09T12:00:00)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Base result directory (default: ./result)",
    )
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir) if args.result_dir else base_dir / "result"
    generate_cache_metrics_report(
        result_base_dir=result_dir,
        date=args.run_date,
        run_id=args.run_id,
    )
