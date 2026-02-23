"""
Example: deploy a HuggingFace classifier via the registry + FastAPI.

Run:
    python examples/hf_model_deploy.py
Then test:
    curl -X POST http://localhost:8000/v1/predict \
         -H "Content-Type: application/json" \
         -d '{"inputs": "I love this product!"}'
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from models.registry import HFPipelineModel, ModelRegistry


async def main():
    registry = ModelRegistry(model_dir=Path("./models/weights"))

    # Manually register a HuggingFace sentiment model
    model = HFPipelineModel(
        name="sentiment",
        task="sentiment-analysis",
        hf_model_id="distilbert-base-uncased-finetuned-sst-2-english",
    )
    await model.load()
    registry.register(model)

    # Direct inference test
    result = await model("This boilerplate is amazing!")
    print("Inference result:", result)

    # List registered models
    print("Models in registry:", registry.list_models())


if __name__ == "__main__":
    asyncio.run(main())
