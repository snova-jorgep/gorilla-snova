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

S3_PREFIX = "fc-so-testing-suite/bfcl_latency_metrics"
CSV_HEADERS = [
    "run_id",
    "date",
    "provider",
    "model",
    "test_category",
    "n_entries",
    "n_calls",
    "avg_latency_s",
    "p50_latency_s",
    "p90_latency_s",
    "p99_latency_s",
    "avg_output_tokens",
    "avg_tpot_ms",
    "avg_throughput_tps",
]


def _flatten_paired(latencies, output_tokens):
    """
    Walk parallel nested structures (both can be scalar, list, or list-of-lists)
    and return flat paired lists [(lat, out_tokens), ...] at the per-step level.
    """
    if isinstance(latencies, list):
        pairs = []
        out_iter = output_tokens if isinstance(output_tokens, list) else [output_tokens] * len(latencies)
        for l, o in zip(latencies, out_iter):
            pairs.extend(_flatten_paired(l, o))
        return pairs
    return [(float(latencies or 0), int(output_tokens or 0))]


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * pct / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def generate_latency_metrics_report(
    result_base_dir: Path,
    date: str,
    run_id: str = None,
) -> Path:
    if run_id is None:
        run_id = str(uuid.uuid4())

    rows = []

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

            category_stats: dict[str, dict] = {}

            for result_file in sorted(model_dir.glob("*_result.json")):
                stem = result_file.stem
                category = stem.replace("BFCL_v3_", "").replace("_result", "")

                stats = category_stats.setdefault(
                    category,
                    {"latencies": [], "outputs": [], "n_entries": 0},
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

                        lat = entry.get("latency", 0)
                        out = entry.get("output_token_count", 0)
                        pairs = _flatten_paired(lat, out)
                        for l, o in pairs:
                            if l > 0:
                                stats["latencies"].append(l)
                                stats["outputs"].append(o)
                        stats["n_entries"] += 1

            for category, stats in category_stats.items():
                lats = sorted(stats["latencies"])
                outs = stats["outputs"]
                n_calls = len(lats)
                if n_calls == 0:
                    continue

                avg_lat = sum(lats) / n_calls
                avg_out = sum(outs) / n_calls

                # TPOT: ms per output token (excluding calls with 0 output)
                tpot_vals = [l / o * 1000 for l, o in zip(lats, outs) if o > 0]
                avg_tpot = sum(tpot_vals) / len(tpot_vals) if tpot_vals else 0

                # Throughput: output tokens / second
                tput_vals = [o / l for l, o in zip(lats, outs) if l > 0]
                avg_tput = sum(tput_vals) / len(tput_vals) if tput_vals else 0

                rows.append({
                    "run_id": run_id,
                    "date": date,
                    "provider": provider,
                    "model": model,
                    "test_category": category,
                    "n_entries": stats["n_entries"],
                    "n_calls": n_calls,
                    "avg_latency_s": round(avg_lat, 4),
                    "p50_latency_s": round(_percentile(lats, 50), 4),
                    "p90_latency_s": round(_percentile(lats, 90), 4),
                    "p99_latency_s": round(_percentile(lats, 99), 4),
                    "avg_output_tokens": round(avg_out, 1),
                    "avg_tpot_ms": round(avg_tpot, 3),
                    "avg_throughput_tps": round(avg_tput, 2),
                })

    if not rows:
        print("No result files found — latency metrics report skipped.")
        return None

    output_dir = base_dir / "results" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latency_metrics.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Latency metrics written to: {output_path}")

    s3_key = f"{S3_PREFIX}/{date}/latency_metrics.csv"
    upload_to_s3(output_path, s3_key)
    print(f"Uploaded to s3://.../{s3_key}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BFCL latency metrics report")
    parser.add_argument("--run-date", type=str, required=True,
                        help="Run date folder name (e.g. 2025-06-09T12:00:00)")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="Base result directory (default: ./result)")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir) if args.result_dir else base_dir / "result"
    generate_latency_metrics_report(
        result_base_dir=result_dir,
        date=args.run_date,
        run_id=args.run_id,
    )
