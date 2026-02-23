"""
Example: run an end-to-end MLOps pipeline + AIOps continuous loop.

Run:
    python examples/run_pipeline.py
"""

from __future__ import annotations

import asyncio
import logging

from pipelines.mlops_pipeline import AIOPsPipeline, MLOpsPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


async def main():
    config = {
        "data_path": "/data/training/v3",
        "model_name": "my_classifier",
        "epochs": 10,
        "drift_threshold": 0.05,
    }

    print("=" * 60)
    print("  MLOps Pipeline")
    print("=" * 60)
    mlops = MLOpsPipeline(config=config)
    run = await mlops.run()
    for step in run.steps:
        icon = "✓" if step.status.name == "SUCCESS" else "✗"
        print(f"  {icon} {step.name:<25} {step.duration_s:.2f}s")
    print(f"\n  Overall: {'SUCCESS' if run.success else 'FAILED'}\n")

    print("=" * 60)
    print("  AIOps Pipeline  ")
    print("=" * 60)
    aiops = AIOPsPipeline(config=config)
    run2 = await aiops.run()
    for step in run2.steps:
        icon = (
            "✓"
            if step.status.name == "SUCCESS"
            else "↷" if step.status.name == "SKIPPED" else "✗"
        )
        print(f"  {icon} {step.name:<25} {step.duration_s:.2f}s")
    print(f"\n  Overall: {'SUCCESS' if run2.success else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
