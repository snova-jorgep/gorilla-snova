import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import List

import pandas as pd
from dotenv import load_dotenv

from config_env import (
    internal_providers_from,
    resolve_config_env,
    row_config_env,
    write_sidecar,
)
from cache_metrics_report import generate_cache_metrics_report
from latency_metrics_report import generate_latency_metrics_report
from generate_subsets import generate_subset
from utils import upload_to_s3

CSV_HEADERS = [
    "run_id",
    "test_suite_name",
    "provider",
    "n_samples",
    "date",
    "DeepSeek-R1-0528",
    "DeepSeek-V3-0324",
    "DeepSeek-V3.1",
    "DeepSeek-V3.1-Terminus",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "Mistral-Large-3",
    "Llama-4-Maverick-17B-128E-Instruct",
    "Llama-4-Scout-17B-16E-Instruct",
    "Meta-Llama-3.1-405B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3.3-70B-Instruct",
    "Qwen3-32B",
    "Qwen3-235B",
    "DeepSeek-V3.2",
    "MiniMax-M2.5",
    "MiniMax-M2.7",
    "Llama-3.3-Swallow-70B-Instruct-v0.4",
    "Gemma-4-31B",
]
DEFAULT_SEED = 42
DATASETS_PREFIX = "BFCL_v3_"

date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

base_dir = Path(__file__).resolve().parent
load_dotenv(base_dir / ".env")
load_dotenv()
results_dir = base_dir / "results"
scores_csv_file_path = results_dir / date / "scores.csv"
scores_csv_file_path.parent.mkdir(parents=True, exist_ok=True)
with open(scores_csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS, delimiter=";")
    writer.writeheader()


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def generate_summary_tables(scores_path: str, summaries_paths, config_env, internal_providers):
    df = pd.read_csv(scores_path, sep=";")
    df = df.dropna(subset=["provider", "test_suite_name"], how="all")

    # Melt the wide format to long format (model columns → rows)
    df_long = df.melt(
        id_vars=["date", "test_suite_name", "provider"],
        value_vars=[
            col
            for col in df.columns
            if col not in ["run_id", "test_suite_name", "provider", "n_samples", "date"]
        ],
        var_name="model",
        value_name="Accuracy",
    )

    for test_type in df["test_suite_name"].dropna().unique():
        test_type = test_type.split(" - ")[-1]
        filtered = df_long[
            df_long["test_suite_name"].str.endswith(f" - {test_type}", na=False)
        ]
        # Convert Accuracy to numeric just in case
        filtered["Accuracy"] = pd.to_numeric(filtered["Accuracy"], errors="coerce")

        # Group by and average (or use first, max, etc.)
        summary = (
            filtered.groupby(["date", "provider", "model"], as_index=False)
            .agg({"Accuracy": "mean"})
            .sort_values(by=["provider", "model"])
        )

        # config_env must be the LAST column: the Athena regex expects it there.
        summary["config_env"] = summary["provider"].apply(
            lambda p: row_config_env(config_env, p, internal_providers)
        )

        # Save to CSV
        summary_filepath = (
            summaries_paths / f"summary_{test_type.lower().replace(' ', '_')}.csv"
        )
        summary.to_csv(summary_filepath, index=False, sep=";")
        print(f"Saved summary to: {summary_filepath}")
        upload_to_s3(
            summary_filepath,
            f"fc-so-testing-suite/gorilla_snova/{date}/{PurePosixPath(summary_filepath).name}",
        )


def run_bfcl_command(command_type, model, test_category, result_dir, score_dir=None):
    result_dir = str(result_dir)
    score_dir = str(score_dir)

    if command_type not in ["generate", "evaluate"]:
        raise ValueError("Invalid command_type. Must be 'generate' or 'evaluate'.")

    command = [
        "bfcl",
        command_type,
        "--model",
        model,
        "--test-category",
        test_category,
        "--result-dir",
        result_dir,
    ]

    if command_type == "evaluate" and score_dir:
        command += ["--score-dir", score_dir]

    try:
        print(f"\nRunning command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"FAILED [{model}] - Return code: {e.returncode}")
        print("Error output:", e.stderr)


def run_models_for_provider(
    provider,
    models,
    run_id,
    test_category,
    model_map,
    lock,
    date,
    concurrency_models: bool = True,
):
    result_path = base_dir / "result" / provider / date
    score_path = base_dir / "score" / provider / date

    def run_model(model, concurrent):
        for category in test_category:
            run_bfcl_command("generate", model, category, result_path)
            run_bfcl_command(
                "evaluate", model, category, result_path, score_dir=score_path
            )
            curr_dict = {
                "run_id": run_id,
                "test_suite_name": f"Berkeley AI Benchmarking - {category}",
                "provider": provider,
                "date": date,
            }
            model_safe = model.replace("/", "_")
            category_score_path = (
                score_path / model_safe / f"BFCL_v3_{category}_score.json"
            )

            try:
                with open(category_score_path, "r") as f:
                    first_line = f.readline()
                    data = json.loads(first_line)
                    curr_dict["n_samples"] = data.get("total_count")
                    curr_dict[model_map[model_safe]] = data.get("accuracy")
                print(f"inserting data: {curr_dict}")
            except FileNotFoundError:
                print(f"Score file not found: {category_score_path}")
            except json.JSONDecodeError:
                print(f"Invalid JSON in file: {category_score_path}")
            except Exception as e:
                print(f"Error reading score for {provider}/{model}: {e}")

            if concurrent:
                with lock:
                    df = pd.read_csv(scores_csv_file_path, sep=";")
                    mask = (df["test_suite_name"] == curr_dict["test_suite_name"]) & (
                        df["provider"] == curr_dict["provider"]
                    )
                    if df[mask].empty:
                        df = pd.concat(
                            [df, pd.DataFrame([curr_dict])], ignore_index=True
                        )
                    else:
                        for key, value in curr_dict.items():
                            df.loc[mask, key] = value
                    df.to_csv(scores_csv_file_path, index=False, sep=";")
                    upload_to_s3(
                        scores_csv_file_path,
                        f"fc-so-testing-suite/gorilla_snova/{date}/{PurePosixPath(scores_csv_file_path).name}",
                    )
            else:
                df = pd.read_csv(scores_csv_file_path, sep=";")
                mask = (df["test_suite_name"] == curr_dict["test_suite_name"]) & (
                    df["provider"] == curr_dict["provider"]
                )
                if df[mask].empty:
                    df = pd.concat([df, pd.DataFrame([curr_dict])], ignore_index=True)
                else:
                    for key, value in curr_dict.items():
                        df.loc[mask, key] = value
                df.to_csv(scores_csv_file_path, index=False, sep=";")
                upload_to_s3(
                    scores_csv_file_path,
                    f"fc-so-testing-suite/gorilla_snova/{date}/{PurePosixPath(scores_csv_file_path).name}",
                )

    if concurrency_models is True:
        # Run all models concurrently
        with concurrent.futures.ThreadPoolExecutor() as model_executor:
            model_futures = [
                model_executor.submit(run_model, model, concurrency_models)
                for model in models
            ]
            concurrent.futures.wait(model_futures)
    else:
        for model in models:
            run_model(model, concurrency_models)


def main(
    test_categories: List[str] = None,
    subset_size: int = 100,
    providers_path: str = None,
    models_mapping_path: str = None,
    config_env: str = None,
):
    resolved_config_env = resolve_config_env(config_env, None)
    internal_providers = internal_providers_from(None)
    print(f"config_env for this run: {resolved_config_env}")
    if providers_path is None:
        providers_path = base_dir / "provider_models.json"
    providers = load_json(providers_path)
    if models_mapping_path is None:
        models_mapping_path = base_dir / "model_map.json"
    model_map = load_json(models_mapping_path)
    if test_categories is None:
        test_categories = [
            "simple",
            "multiple",
            "live_parallel",
            "multi_turn_base",
            "parallel_multiple",
            "multi_turn_long_context",
        ]
    run_id = str(uuid.uuid4())
    lock = Lock()

    for category in test_categories:
        try:
            generate_subset(
                dataset_name=f"{DATASETS_PREFIX}{category}", subset_size=subset_size, seed=DEFAULT_SEED
            )
        except Exception as e:
            raise Exception(f"Subset generation Error output: {category}: {e}")
            return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_models_for_provider,
                provider,
                models,
                run_id,
                test_categories,
                model_map,
                lock,
                date,
            )
            for provider, models in providers.items()
        ]
        concurrent.futures.wait(futures)

    generate_summary_tables(
        scores_csv_file_path,
        scores_csv_file_path.parent,  # summaries belong in results/<date>/, next to scores.csv
        resolved_config_env,
        internal_providers,
    )

    # Persist config_env next to the run so later report scripts can recover it.
    write_sidecar(scores_csv_file_path.parent, resolved_config_env, {})
    generate_cache_metrics_report(
        result_base_dir=base_dir / "result",
        date=date,
        run_id=run_id,
    )
    generate_latency_metrics_report(
        result_base_dir=base_dir / "result",
        date=date,
        run_id=run_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run berkeley benchmarks")
    parser.add_argument(
        "--test-categories",
        nargs="+",
        default=None,
        help="List of test categories to use (space-separated). Example: --test-categories simple multiple",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Size of the data subset to generate for each category",
    )
    parser.add_argument(
        "--providers-path",
        type=str,
        default=None,
        help="Path to JSON file with provider-model mappings",
    )
    parser.add_argument(
        "--models-mapping-path",
        type=str,
        default=None,
        help="Path to JSON file with model mapping",
    )
    parser.add_argument(
        "--config-env",
        type=str,
        default=None,
        help="Config env id for this run, e.g. sn50-tp8dp2-mtp. "
             "Defaults to default_config_env from config_envs.yaml (cloud-prod). "
             "Must match the endpoint set in "
             "bfcl_eval/model_handler/api_inference/openai_compatible.py::_init_client",
    )
    args = parser.parse_args()

    start_time = time.time()
    main(
        test_categories=args.test_categories,
        subset_size=args.subset_size,
        providers_path=args.providers_path,
        models_mapping_path=args.models_mapping_path,
        config_env=args.config_env,
    )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
