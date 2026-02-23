"""
MLOps + AIOps pipeline boilerplate.

MLOpsPipeline: train → evaluate → register → deploy → monitor
AIOPsPipeline: ingest live data → detect drift → trigger retraining → re-deploy
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Step result ───────────────────────────────────────────────────────


class StepStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StepResult:
    name: str
    status: StepStatus
    duration_s: float = 0.0
    output: Any = None
    error: Optional[str] = None


@dataclass
class PipelineRun:
    pipeline_name: str
    run_id: str
    steps: List[StepResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return all(
            s.status in (StepStatus.SUCCESS, StepStatus.SKIPPED) for s in self.steps
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline": self.pipeline_name,
            "success": self.success,
            "steps": [
                {"name": s.name, "status": s.status.name, "duration_s": s.duration_s}
                for s in self.steps
            ],
        }


# ── Core pipeline engine ──────────────────────────────────────────────


class Pipeline:
    """
    Sequential async pipeline with named steps.

    Usage:
        p = Pipeline("my-pipeline")
        p.add_step("preprocess", my_preprocess_fn)
        p.add_step("train",      my_train_fn)
        run = await p.run(context={...})
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: List[tuple[str, Callable]] = []

    def add_step(self, name: str, fn: Callable) -> "Pipeline":
        self._steps.append((name, fn))
        return self

    async def run(self, context: Dict[str, Any]) -> PipelineRun:
        run_id = f"{self.name}-{int(time.time())}"
        run = PipelineRun(
            pipeline_name=self.name, run_id=run_id, metadata=context.copy()
        )
        logger.info("Pipeline '%s' started (run_id=%s)", self.name, run_id)

        for step_name, fn in self._steps:
            t0 = time.perf_counter()
            result = StepResult(name=step_name, status=StepStatus.RUNNING)
            try:
                logger.info("  [%s] running...", step_name)
                if asyncio.iscoroutinefunction(fn):
                    output = await fn(context)
                else:
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(None, fn, context)
                result.output = output
                result.status = StepStatus.SUCCESS
                if output is not None:
                    context[step_name] = output
            except Exception as exc:
                result.status = StepStatus.FAILED
                result.error = str(exc)
                logger.exception("  [%s] FAILED: %s", step_name, exc)
                run.steps.append(result)
                break
            finally:
                result.duration_s = round(time.perf_counter() - t0, 3)

            logger.info("  [%s] done in %.2fs", step_name, result.duration_s)
            run.steps.append(result)

        logger.info(
            "Pipeline '%s' %s.", self.name, "SUCCEEDED" if run.success else "FAILED"
        )
        return run


# ── MLOps Pipeline ────────────────────────────────────────────────────


class MLOpsPipeline:
    """
    End-to-end MLOps pipeline:
        data_validation → preprocessing → training → evaluation
        → model_registry → deployment → monitoring_setup
    Swap each step with your real implementation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._pipeline = Pipeline("mlops")
        self._build()

    def _build(self) -> None:
        self._pipeline.add_step("data_validation", self._data_validation)
        self._pipeline.add_step("preprocessing", self._preprocessing)
        self._pipeline.add_step("training", self._training)
        self._pipeline.add_step("evaluation", self._evaluation)
        self._pipeline.add_step("model_registry", self._model_registry)
        self._pipeline.add_step("deployment", self._deployment)
        self._pipeline.add_step("monitoring_setup", self._monitoring_setup)

    # ── Step implementations (replace with real logic) ─────────────
    async def _data_validation(self, ctx: Dict) -> Dict:
        logger.info("Validating dataset at %s", ctx.get("data_path"))
        # TODO: great_expectations / pandera schema checks
        return {"data_valid": True, "n_samples": 10000}

    async def _preprocessing(self, ctx: Dict) -> Dict:
        logger.info("Preprocessing data...")
        # TODO: feature engineering, normalization, splits
        return {"train_path": "/tmp/train.pt", "val_path": "/tmp/val.pt"}

    async def _training(self, ctx: Dict) -> Dict:
        logger.info("Training model...")
        # TODO: instantiate Trainer from 03-training boilerplate
        return {"checkpoint_path": "/tmp/model.pt", "train_loss": 0.21}

    async def _evaluation(self, ctx: Dict) -> Dict:
        logger.info("Evaluating model...")
        # TODO: compute metrics, compare with baseline
        return {"accuracy": 0.94, "f1": 0.93, "passed_gate": True}

    async def _model_registry(self, ctx: Dict) -> Dict:
        if not ctx.get("evaluation", {}).get("passed_gate", False):
            raise ValueError("Model did not pass quality gate — aborting registration.")
        logger.info("Registering model in MLflow / registry...")
        # TODO: mlflow.register_model(...)
        return {"model_version": "v1", "registry_uri": "models:/my_model/1"}

    async def _deployment(self, ctx: Dict) -> Dict:
        logger.info(
            "Deploying model version %s...",
            ctx.get("model_registry", {}).get("model_version"),
        )
        # TODO: copy weights, reload registry, canary / blue-green logic
        return {"endpoint": "/v1/predict", "status": "active"}

    async def _monitoring_setup(self, ctx: Dict) -> Dict:
        logger.info("Setting up drift monitors and alerting...")
        # TODO: Evidently AI / WhyLogs / custom monitor registration
        return {"monitor": "active", "alert_threshold": 0.05}

    async def run(self) -> PipelineRun:
        return await self._pipeline.run(self.config.copy())


# ── AIOps Pipeline ────────────────────────────────────────────────────


class AIOPsPipeline:
    """
    Continuous AIOps loop:
        ingest_live_data → drift_detection → [conditional] retraining_trigger
        → retraining → re_deployment → notification
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._pipeline = Pipeline("aiops")
        self._build()

    def _build(self) -> None:
        self._pipeline.add_step("ingest_live_data", self._ingest_live_data)
        self._pipeline.add_step("drift_detection", self._drift_detection)
        self._pipeline.add_step("retraining_trigger", self._retraining_trigger)
        self._pipeline.add_step("retraining", self._retraining)
        self._pipeline.add_step("re_deployment", self._re_deployment)
        self._pipeline.add_step("notification", self._notification)

    async def _ingest_live_data(self, ctx: Dict) -> Dict:
        logger.info("Ingesting live production data...")
        # TODO: pull from feature store / data lake / stream
        return {"n_new_samples": 500, "window": "1h"}

    async def _drift_detection(self, ctx: Dict) -> Dict:
        logger.info("Running drift detection...")
        # TODO: KS test, PSI, MMD via Evidently / custom
        drift_detected = True  # placeholder
        drift_score = 0.12
        logger.info("Drift score: %.3f | Detected: %s", drift_score, drift_detected)
        return {"drift_detected": drift_detected, "drift_score": drift_score}

    async def _retraining_trigger(self, ctx: Dict) -> Dict:
        drift_info = ctx.get("drift_detection", {})
        threshold = self.config.get("drift_threshold", 0.05)
        should_retrain = drift_info.get("drift_score", 0) > threshold
        logger.info("Retraining needed: %s", should_retrain)
        return {"should_retrain": should_retrain}

    async def _retraining(self, ctx: Dict) -> Dict:
        if not ctx.get("retraining_trigger", {}).get("should_retrain"):
            logger.info("No retraining needed — skipping.")
            return {"skipped": True}
        logger.info("Triggering incremental retraining...")
        # TODO: kick off MLOpsPipeline or Airflow/Prefect DAG
        return {"new_checkpoint": "/tmp/model_v2.pt", "improved": True}

    async def _re_deployment(self, ctx: Dict) -> Dict:
        if ctx.get("retraining", {}).get("skipped"):
            return {"skipped": True}
        logger.info("Re-deploying updated model...")
        return {"status": "deployed", "version": "v2"}

    async def _notification(self, ctx: Dict) -> Dict:
        logger.info("Sending pipeline completion notification...")
        # TODO: Slack / PagerDuty / email
        return {"notified": True}

    async def run(self) -> PipelineRun:
        return await self._pipeline.run(self.config.copy())
