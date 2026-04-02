"""Experiment tracker — auto-updates experiment.yaml after each training stage."""

from datetime import datetime
from pathlib import Path

import yaml


def find_experiment_dir(config_path: str) -> Path | None:
    """Find the experiment directory from a config path.

    Config paths look like: experiments/mistral-nemo-12b-v1/configs/sft_config.yaml
    The experiment dir is: experiments/mistral-nemo-12b-v1/
    """
    config_path = Path(config_path)
    # Walk up from config file: configs/ → experiment dir
    if config_path.parent.name == "configs" and config_path.parent.parent.parent.name == "experiments":
        return config_path.parent.parent
    return None


def update_experiment_status(config_path: str, stage: str, status: str):
    """Update experiment.yaml status after a training stage completes.

    Args:
        config_path: Path to the config YAML that was used for training
        stage: "sft", "raft", or "dpo"
        status: "sft_done", "raft_done", "dpo_done", "completed", or "failed"
    """
    exp_dir = find_experiment_dir(config_path)
    if exp_dir is None:
        return  # Not running from an experiment directory (e.g., using template configs)

    manifest_path = exp_dir / "experiment.yaml"
    if not manifest_path.exists():
        return

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    manifest["status"] = status
    manifest[f"{stage}_completed"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    print(f"[tracker] Updated {manifest_path}: status → {status}")


def record_training_metrics(config_path: str, stage: str, metrics: dict):
    """Write key metrics to the results YAML after training completes.

    Args:
        config_path: Path to the config YAML
        stage: "sft", "raft", or "dpo"
        metrics: Dict of metric values to record
    """
    exp_dir = find_experiment_dir(config_path)
    if exp_dir is None:
        return

    results_path = exp_dir / "results" / f"{stage}_results.yaml"
    if not results_path.exists():
        return

    with open(results_path) as f:
        results = yaml.safe_load(f) or {}

    # Update date and metrics
    results["date"] = datetime.now().strftime("%Y-%m-%d")
    if "metrics" not in results:
        results["metrics"] = {}
    results["metrics"].update(metrics)

    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print(f"[tracker] Updated {results_path} with {list(metrics.keys())}")
