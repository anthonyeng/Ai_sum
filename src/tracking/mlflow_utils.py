"""
MLflow experiment tracking utilities.

Provides a clean interface for logging training runs, metrics, parameters,
and model artifacts. Falls back gracefully if MLflow is not installed.

Usage:
    from src.tracking.mlflow_utils import ExperimentTracker

    tracker = ExperimentTracker("video_summarization")
    tracker.start_run("xgboost_v1")
    tracker.log_params({"n_estimators": 400, "max_depth": 6})
    tracker.log_metrics({"mse": 0.05, "r2": 0.84}, step=10)
    tracker.log_model("outputs/models/xgboost.pkl")
    tracker.end_run()
"""

import os
import json
from typing import Optional

# MLflow tracking URI — local file store (no server needed)
_TRACKING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "mlruns",
)

_mlflow = None


def _load_mlflow():
    """Lazy-load MLflow."""
    global _mlflow
    if _mlflow is not None:
        return _mlflow
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file://{_TRACKING_DIR}")
        _mlflow = mlflow
        return mlflow
    except ImportError:
        return None


class ExperimentTracker:
    """
    Wrapper around MLflow for experiment tracking.

    Falls back to JSON file logging if MLflow is not installed.
    """

    def __init__(self, experiment_name: str = "ai_video_summarizer"):
        self.experiment_name = experiment_name
        self.mlflow = _load_mlflow()
        self._run = None
        self._fallback_log = {}
        self._run_name = None

        if self.mlflow:
            self.mlflow.set_experiment(experiment_name)

    @property
    def is_active(self) -> bool:
        return self._run is not None or (not self.mlflow and self._run_name is not None)

    def start_run(self, run_name: str, tags: Optional[dict] = None):
        """Start a new tracking run."""
        self._run_name = run_name

        if self.mlflow:
            self._run = self.mlflow.start_run(run_name=run_name)
            if tags:
                self.mlflow.set_tags(tags)
        else:
            self._fallback_log = {
                "run_name": run_name,
                "tags": tags or {},
                "params": {},
                "metrics": {},
            }

    def log_params(self, params: dict):
        """Log hyperparameters."""
        if self.mlflow and self._run:
            self.mlflow.log_params(params)
        else:
            self._fallback_log.setdefault("params", {}).update(params)

    def log_param(self, key: str, value):
        """Log a single parameter."""
        self.log_params({key: value})

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics (optionally at a specific step/epoch)."""
        if self.mlflow and self._run:
            self.mlflow.log_metrics(metrics, step=step)
        else:
            for k, v in metrics.items():
                self._fallback_log.setdefault("metrics", {}).setdefault(k, []).append(
                    {"value": v, "step": step}
                )

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)

    def log_model(self, model_path: str, artifact_name: Optional[str] = None):
        """Log a model artifact."""
        if self.mlflow and self._run:
            name = artifact_name or os.path.basename(model_path)
            self.mlflow.log_artifact(model_path, artifact_path="models")
        else:
            self._fallback_log["model_path"] = model_path

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log any file as an artifact."""
        if self.mlflow and self._run:
            self.mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        if self.mlflow and self._run:
            self.mlflow.end_run(status=status)
            self._run = None
        elif self._fallback_log:
            # Save fallback log to JSON
            log_dir = os.path.join(_TRACKING_DIR, "fallback_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{self._run_name}.json")
            with open(log_path, "w") as f:
                json.dump(self._fallback_log, f, indent=2, default=str)
            self._fallback_log = {}

        self._run_name = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.is_active:
            self.end_run(status="FAILED" if args[0] else "FINISHED")


def compare_runs(experiment_name: str = "ai_video_summarizer") -> Optional[dict]:
    """
    Load and compare all runs from an experiment.

    Returns:
        dict with run comparisons, or None if MLflow not available.
    """
    mlflow = _load_mlflow()
    if not mlflow:
        return None

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return {"error": f"Experiment '{experiment_name}' not found"}

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_loss ASC"],
        )

        if runs.empty:
            return {"error": "No runs found"}

        results = []
        for _, run in runs.iterrows():
            entry = {
                "run_name": run.get("tags.mlflow.runName", "unknown"),
                "status": run.get("status", "unknown"),
            }
            # Extract metrics
            for col in runs.columns:
                if col.startswith("metrics."):
                    metric_name = col.replace("metrics.", "")
                    val = run[col]
                    if not (isinstance(val, float) and val != val):  # skip NaN
                        entry[metric_name] = round(float(val), 4) if isinstance(val, (int, float)) else val
            # Extract params
            for col in runs.columns:
                if col.startswith("params."):
                    param_name = col.replace("params.", "")
                    entry[param_name] = run[col]
            results.append(entry)

        return {
            "experiment": experiment_name,
            "num_runs": len(results),
            "runs": results,
        }
    except Exception as e:
        return {"error": str(e)}
