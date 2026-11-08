"""Low-latency model serving with batching, caching, and monitoring."""

import asyncio
import logging
import time
import pickle
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict
import numpy as np
import xgboost as xgb

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from core.schemas import (
    PredictionRequest, PredictionResponse, FeatureVector,
    ServingEndpoint, ServingStatus, ServingConfig, ModelSpec, ModelType
)
from training.models import BaseModel, create_model
from features.online_store import OnlineFeatureStore, MockOnlineFeatureStore

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Batched prediction request."""
    requests: List[PredictionRequest]
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)


class ModelLoader:
    """Load and manage models for serving."""

    def __init__(self, model_dir: str = "artifacts/models"):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, BaseModel] = {}
        self.model_specs: Dict[str, ModelSpec] = {}
        self.feature_pipelines: Dict[str, Any] = {}
        self.online_store: Optional[OnlineFeatureStore] = None

    def set_online_store(self, store: OnlineFeatureStore):
        self.online_store = store

    def load_model(self, model_id: str, version: str = "latest") -> BaseModel:
        """Load model from disk."""
        cache_key = f"{model_id}:{version}"

        if cache_key in self.models:
            return self.models[cache_key]

        # Find model artifact
        if version == "latest":
            # Find latest version directory
            model_dirs = list(self.model_dir.glob(f"{model_id}_*"))
            if not model_dirs:
                raise FileNotFoundError(f"No model found for {model_id}")
            model_path = max(model_dirs, key=lambda p: p.stat().st_mtime)
        else:
            model_path = self.model_dir / f"{model_id}_{version}"

        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Load model spec
        spec_path = model_path / "model_spec.json"
        if spec_path.exists():
            with open(spec_path, 'r') as f:
                spec_data = json.load(f)
            model_spec = ModelSpec(**spec_data)
        else:
            # Try to find .pkl file
            pkl_files = list(model_path.glob("*.pkl")) + list(model_path.glob("*.xgb"))
            if not pkl_files:
                raise FileNotFoundError(f"No model artifact found in {model_path}")
            model_spec = None

        # Load model
        if model_spec:
            model = create_model(model_spec)
            # Load actual model weights
            artifact_path = model_path / "model.xgb"
            if artifact_path.exists():
                model.model = xgb.Booster()
                model.model.load_model(str(artifact_path))
                model.is_fitted = True

            # Load feature pipeline
            pipeline_path = model_path / "feature_pipeline.pkl"
            if pipeline_path.exists():
                with open(pipeline_path, 'rb') as f:
                    model.feature_pipeline = pickle.load(f)
        else:
            # Load from pickle directly
            with open(pkl_files[0], 'rb') as f:
                data = pickle.load(f)
            model = data['model'] if isinstance(data, dict) else data
            if hasattr(model, 'model_spec'):
                model_spec = model.model_spec
            else:
                raise ValueError("Model spec not found")

        self.models[cache_key] = model
        self.model_specs[cache_key] = model_spec

        logger.info(f"Loaded model: {cache_key}")
        return model

    def unload_model(self, model_id: str, version: str = "latest"):
        """Unload model from memory."""
        cache_key = f"{model_id}:{version}"
        if cache_key in self.models:
            del self.models[cache_key]
            del self.model_specs[cache_key]
            logger.info(f"Unloaded model: {cache_key}")

    def get_model(self, model_id: str, version: str = "latest") -> BaseModel:
        """Get loaded model or load it."""
        cache_key = f"{model_id}:{version}"
        if cache_key not in self.models:
            return self.load_model(model_id, version)
        return self.models[cache_key]


class PredictionCache:
    """Simple LRU cache for predictions."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = {}  # key -> (response, timestamp)
        self.access_order: List[str] = []

    def _make_key(self, request: PredictionRequest) -> str:
        """Create cache key from request."""
        # Use model_id, version, and entity IDs
        entity_ids = "-".join(sorted([e.entity_id for e in request.entities]))
        return f"{request.model_id}:{request.model_version}:{entity_ids}"

    def get(self, request: PredictionRequest) -> Optional[PredictionResponse]:
        """Get cached prediction."""
        key = self._make_key(request)

        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recent)
                self.access_order.remove(key)
                self.access_order.append(key)
                response.cache_hit = True
                return response
            else:
                # Expired
                del self.cache[key]
                self.access_order.remove(key)

        return None

    def set(self, request: PredictionRequest, response: PredictionResponse):
        """Cache prediction."""
        key = self._make_key(request)

        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict LRU
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = (response, time.time())
        self.access_order.append(key)

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache stats."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hit_rate": getattr(self, '_hit_rate', 0.0)
        }


class BatchProcessor:
    """Batch prediction processor for throughput optimization."""

    def __init__(
        self,
        model_loader: ModelLoader,
        max_batch_size: int = 32,
        max_wait_ms: int = 10
    ):
        self.model_loader = model_loader
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start batch processor."""
        self.running = True
        self.worker_task = asyncio.create_task(self._process_batches())
        logger.info("Batch processor started")

    async def stop(self):
        """Stop batch processor."""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Batch processor stopped")

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Submit prediction request for batch processing."""
        future = asyncio.get_event_loop().create_future()
        batch_req = BatchRequest(requests=[request], future=future)
        await self.batch_queue.put(batch_req)
        return await future

    async def _process_batches(self):
        """Process batches from queue."""
        while self.running:
            try:
                # Collect batch
                batch_requests = []
                futures = []

                # Get first request (wait)
                batch_req = await asyncio.wait_for(
                    self.batch_queue.get(),
                    timeout=self.max_wait_ms / 1000
                )
                batch_requests.extend(batch_req.requests)
                futures.append(batch_req.future)

                # Collect more requests up to batch size
                while len(batch_requests) < self.max_batch_size:
                    try:
                        batch_req = self.batch_queue.get_nowait()
                        batch_requests.extend(batch_req.requests)
                        futures.append(batch_req.future)
                    except asyncio.QueueEmpty:
                        break

                # Process batch
                if batch_requests:
                    await self._process_batch(batch_requests, futures)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Resolve futures with error
                for f in futures:
                    if not f.done():
                        f.set_exception(e)

    async def _process_batch(
        self,
        requests: List[PredictionRequest],
        futures: List[asyncio.Future]
    ):
        """Process a batch of requests."""
        start_time = time.time()

        try:
            # Group by model
            model_groups = defaultdict(list)
            for i, req in enumerate(requests):
                key = f"{req.model_id}:{req.model_version}"
                model_groups[key].append((i, req))

            # Process each model group
            results = [None] * len(requests)

            for model_key, group in model_groups.items():
                model_id, version = model_key.split(":")
                model = self.model_loader.get_model(model_id, version)

                # Prepare batch input
                batch_entities = []
                batch_contexts = []
                for idx, req in group:
                    batch_entities.extend(req.entities)
                    batch_contexts.append(req.context)

                # Convert to DataFrame for prediction
                # Simplified - in production, use proper batch prediction
                predictions = []
                for req in [g[1] for g in group]:
                    X = model._request_to_dataframe(req)
                    preds = model.predict_proba(X)
                    predictions.append([{"score": float(p)} for p in preds])

                # Create responses
                for (idx, req), preds in zip(group, predictions):
                    latency = (time.time() - start_time) * 1000
                    results[idx] = PredictionResponse(
                        request_id=req.request_id,
                        model_id=req.model_id,
                        model_version=req.model_version,
                        predictions=preds,
                        latency_ms=latency
                    )

            # Resolve futures
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            for future in futures:
                if not future.done():
                    future.set_exception(e)


class ModelServer:
    """Main model serving server."""

    def __init__(self, config: ServingConfig):
        self.config = config
        self.model_loader = ModelLoader()
        self.cache = PredictionCache(
            max_size=10000,
            ttl_seconds=config.cache_ttl_seconds
        ) if config.enable_caching else None
        self.batch_processor = BatchProcessor(
            self.model_loader,
            max_batch_size=config.batch_size,
            max_wait_ms=10
        ) if config.enable_batching else None
        self.app = self._create_app()
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.latency_sum = 0.0

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            if self.batch_processor:
                await self.batch_processor.start()
            logger.info(f"Model server started on {self.config.host}:{self.config.port}")
            yield
            # Shutdown
            if self.batch_processor:
                await self.batch_processor.stop()
            logger.info("Model server stopped")

        app = FastAPI(
            title="ML Model Serving",
            version="1.0.0",
            lifespan=lifespan
        )

        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            return await self._handle_predict(request)

        @app.post("/predict/batch", response_model=List[PredictionResponse])
        async def predict_batch(requests: List[PredictionRequest]):
            return await self._handle_predict_batch(requests)

        @app.get("/health")
        async def health():
            return await self._health_check()

        @app.get("/metrics")
        async def metrics():
            return self._get_metrics()

        @app.get("/models/{model_id}/versions/{version}")
        async def get_model_info(model_id: str, version: str):
            return self._get_model_info(model_id, version)

        @app.post("/models/{model_id}/versions/{version}/load")
        async def load_model(model_id: str, version: str):
            return self._load_model(model_id, version)

        @app.post("/models/{model_id}/versions/{version}/unload")
        async def unload_model(model_id: str, version: str):
            return self._unload_model(model_id, version)

        @app.post("/cache/clear")
        async def clear_cache():
            if self.cache:
                self.cache.clear()
            return {"status": "cleared"}

        return app

    async def _handle_predict(self, request: PredictionRequest) -> PredictionResponse:
        """Handle single prediction request."""
        start_time = time.time()
        self.request_count += 1

        try:
            # Check cache
            if self.cache:
                cached = self.cache.get(request)
                if cached:
                    return cached

            # Get model
            model = self.model_loader.get_model(request.model_id, request.model_version)

            # Run prediction
            if self.batch_processor:
                response = await self.batch_processor.predict(request)
            else:
                # Direct prediction
                X = model._request_to_dataframe(request)
                preds = model.predict_proba(X)
                response = PredictionResponse(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    model_version=request.model_version,
                    predictions=[{"score": float(p)} for p in preds],
                    latency_ms=(time.time() - start_time) * 1000
                )

            # Cache response
            if self.cache:
                self.cache.set(request, response)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.latency_sum += latency

            return response

        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _handle_predict_batch(
        self,
        requests: List[PredictionRequest]
    ) -> List[PredictionResponse]:
        """Handle batch prediction request."""
        start_time = time.time()
        self.request_count += len(requests)

        try:
            if self.batch_processor:
                # Submit all to batch processor
                tasks = [self.batch_processor.predict(req) for req in requests]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions
                results = []
                for i, resp in enumerate(responses):
                    if isinstance(resp, Exception):
                        logger.error(f"Batch item {i} failed: {resp}")
                        results.append(PredictionResponse(
                            request_id=requests[i].request_id,
                            model_id=requests[i].model_id,
                            model_version=requests[i].model_version,
                            predictions=[],
                            latency_ms=(time.time() - start_time) * 1000,
                            cache_hit=False
                        ))
                    else:
                        results.append(resp)

                return results
            else:
                # Process sequentially
                return [await self._handle_predict(req) for req in requests]

        except Exception as e:
            self.error_count += len(requests)
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - self.start_time,
            "models_loaded": len(self.model_loader.models),
            "cache_enabled": self.config.enable_caching,
            "batching_enabled": self.config.enable_batching
        }

    def _get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        avg_latency = self.latency_sum / max(1, self.request_count)
        error_rate = self.error_count / max(1, self.request_count)

        metrics = {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency,
            "uptime_seconds": time.time() - self.start_time,
            "models_loaded": len(self.model_loader.models),
            "cache_stats": self.cache.stats() if self.cache else {}
        }

        return metrics

    def _get_model_info(self, model_id: str, version: str) -> Dict[str, Any]:
        """Get model information."""
        cache_key = f"{model_id}:{version}"

        if cache_key in self.model_loader.model_specs:
            spec = self.model_loader.model_specs[cache_key]
            return {
                "model_id": model_id,
                "version": version,
                "name": spec.name,
                "model_type": spec.model_type.value,
                "framework": spec.framework,
                "metrics": spec.metrics,
                "loaded": True
            }

        # Try to load
        try:
            model = self.model_loader.load_model(model_id, version)
            spec = self.model_loader.model_specs[cache_key]
            return {
                "model_id": model_id,
                "version": version,
                "name": spec.name,
                "model_type": spec.model_type.value,
                "framework": spec.framework,
                "metrics": spec.metrics,
                "loaded": True
            }
        except Exception as e:
            return {
                "model_id": model_id,
                "version": version,
                "loaded": False,
                "error": str(e)
            }

    def _load_model(self, model_id: str, version: str) -> Dict[str, Any]:
        """Load model endpoint."""
        try:
            model = self.model_loader.load_model(model_id, version)
            return {"status": "loaded", "model_id": model_id, "version": version}
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    def _unload_model(self, model_id: str, version: str) -> Dict[str, Any]:
        """Unload model endpoint."""
        self.model_loader.unload_model(model_id, version)
        return {"status": "unloaded", "model_id": model_id, "version": version}

    def run(self):
        """Run the server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=1,  # Use 1 worker with async batching
            log_level="info"
        )


def create_serving_config(
    model_id: str,
    model_version: str = "latest",
    host: str = "0.0.0.0",
    port: int = 8080,
    **kwargs
) -> ServingConfig:
    """Create serving configuration."""
    return ServingConfig(
        model_id=model_id,
        model_version=model_version,
        host=host,
        port=port,
        **kwargs
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = create_serving_config(
        model_id="ranker_v1",
        model_version="1.0.0",
        port=8080
    )

    server = ModelServer(config)
    print("Model server created. Run with: server.run()")