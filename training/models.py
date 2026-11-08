"""Model implementations for ranking, recommendation, search, and ads."""

import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from abc import ABC, abstractmethod
import pickle

from core.schemas import ModelSpec, ModelType, FeatureVector, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all models."""

    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec
        self.model = None
        self.feature_pipeline = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Batch prediction for serving."""
        responses = []
        for req in requests:
            # Convert request to DataFrame
            X = self._request_to_dataframe(req)
            preds = self.predict_proba(X)
            responses.append(PredictionResponse(
                request_id=req.request_id,
                model_id=self.model_spec.model_id,
                model_version=self.model_spec.version,
                predictions=[{"score": float(p)} for p in preds],
                latency_ms=0.0  # Will be filled by serving layer
            ))
        return responses

    def _request_to_dataframe(self, request: PredictionRequest) -> pd.DataFrame:
        """Convert prediction request to DataFrame."""
        rows = []
        for entity in request.entities:
            row = entity.features.copy()
            row.update(request.context)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_pipeline': self.feature_pipeline,
                'model_spec': self.model_spec,
                'is_fitted': self.is_fitted
            }, f)

    @classmethod
    def load(cls, path: str, model_spec: ModelSpec = None) -> 'BaseModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(model_spec or data['model_spec'])
        instance.model = data['model']
        instance.feature_pipeline = data['feature_pipeline']
        instance.is_fitted = data['is_fitted']
        return instance


class RankingModel(BaseModel):
    """Ranking model for search/feed ranking."""

    def __init__(
        self,
        model_spec: ModelSpec,
        params: Dict[str, Any] = None,
        feature_pipeline = None
    ):
        super().__init__(model_spec)
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "seed": 42,
            "nthread": -1,
            "verbosity": 1
        }
        self.feature_pipeline = feature_pipeline
        self.num_boost_round = 1000
        self.early_stopping_rounds = 50

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        **kwargs
    ) -> 'RankingModel':
        """Train ranking model."""
        # Transform features if pipeline exists
        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)
            if X_val is not None:
                X_val = self.feature_pipeline.transform(X_val)

        dtrain = xgb.DMatrix(X, label=y)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, "train"), (dval, "val")]
        else:
            evals = [(dtrain, "train")]

        evals_result = {}

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            evals_result=evals_result,
            verbose_eval=kwargs.get('verbose_eval', 50)
        )

        self.is_fitted = True
        logger.info(f"Ranking model trained. Best iteration: {self.model.best_iteration}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict click probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability (same as predict for binary classification)."""
        return self.predict(X)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        importance = self.model.get_score(importance_type=importance_type)
        total = sum(importance.values())
        return {k: v/total for k, v in importance.items()}


class RecommendationModel(BaseModel):
    """Recommendation model (implicit feedback / collaborative filtering style)."""

    def __init__(
        self,
        model_spec: ModelSpec,
        params: Dict[str, Any] = None,
        feature_pipeline = None,
        embedding_dim: int = 64
    ):
        super().__init__(model_spec)
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "seed": 42,
            "nthread": -1,
            "verbosity": 1
        }
        self.feature_pipeline = feature_pipeline
        self.embedding_dim = embedding_dim
        self.num_boost_round = 500
        self.early_stopping_rounds = 30
        self.user_embeddings = {}
        self.item_embeddings = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        **kwargs
    ) -> 'RecommendationModel':
        """Train recommendation model."""
        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)
            if X_val is not None:
                X_val = self.feature_pipeline.transform(X_val)

        dtrain = xgb.DMatrix(X, label=y)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, "train"), (dval, "val")]
        else:
            evals = [(dtrain, "train")]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=kwargs.get('verbose_eval', 50)
        )

        self.is_fitted = True

        # Extract embeddings if available in features
        self._extract_embeddings(X)

        return self

    def _extract_embeddings(self, X: pd.DataFrame) -> None:
        """Extract user/item embeddings from feature columns."""
        emb_cols = [c for c in X.columns if c.endswith('_emb_')]
        if not emb_cols:
            return

        # Group by prefix (user_emb_, item_emb_)
        for col in emb_cols:
            prefix = col.split('_emb_')[0]
            if prefix not in self.user_embeddings and prefix not in self.item_embeddings:
                # Get all embedding columns for this prefix
                prefix_cols = [c for c in emb_cols if c.startswith(prefix + '_emb_')]
                if prefix_cols:
                    # Store mean embedding per entity (simplified)
                    pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict interaction probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def recommend(
        self,
        user_features: Dict[str, Any],
        candidate_items: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Generate top-K recommendations for a user."""
        # Create feature vectors for each candidate
        rows = []
        for item in candidate_items:
            row = user_features.copy()
            row.update(item)
            rows.append(row)

        X = pd.DataFrame(rows)
        scores = self.predict_proba(X)

        # Get top-K
        item_ids = [item.get('item_id', f'item_{i}') for i, item in enumerate(candidate_items)]
        scored = list(zip(item_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]


class SearchRerankerModel(BaseModel):
    """Search result reranker model."""

    def __init__(
        self,
        model_spec: ModelSpec,
        params: Dict[str, Any] = None,
        feature_pipeline = None,
        listwise: bool = False
    ):
        super().__init__(model_spec)
        self.params = params or {
            "objective": "rank:pairwise",  # or rank:ndcg for listwise
            "eval_metric": "ndcg@10",
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "nthread": -1,
            "verbosity": 1
        }
        self.feature_pipeline = feature_pipeline
        self.listwise = listwise
        self.num_boost_round = 1000
        self.early_stopping_rounds = 50

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray = None,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        groups_val: np.ndarray = None,
        **kwargs
    ) -> 'SearchRerankerModel':
        """Train reranker with group information for listwise ranking."""
        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)
            if X_val is not None:
                X_val = self.feature_pipeline.transform(X_val)

        dtrain = xgb.DMatrix(X, label=y)
        if groups is not None:
            dtrain.set_group(groups)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            if groups_val is not None:
                dval.set_group(groups_val)
            evals = [(dtrain, "train"), (dval, "val")]
        else:
            evals = [(dtrain, "train")]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=kwargs.get('verbose_eval', 50)
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, groups: np.ndarray = None) -> np.ndarray:
        """Predict relevance scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)

        dtest = xgb.DMatrix(X)
        if groups is not None:
            dtest.set_group(groups)

        return self.model.predict(dtest)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def rerank(
        self,
        query_features: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Rerank search results."""
        rows = []
        for cand in candidates:
            row = query_features.copy()
            row.update(cand)
            rows.append(row)

        X = pd.DataFrame(rows)
        scores = self.predict_proba(X)

        # Sort by score
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = float(scores[i])

        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        if top_k:
            candidates = candidates[:top_k]

        return candidates


class CTRPredictorModel(BaseModel):
    """Click-through rate prediction model."""

    def __init__(
        self,
        model_spec: ModelSpec,
        params: Dict[str, Any] = None,
        feature_pipeline = None,
        calibration: bool = True
    ):
        super().__init__(model_spec)
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "scale_pos_weight": 1,  # Will be set based on data
            "seed": 42,
            "nthread": -1,
            "verbosity": 1
        }
        self.feature_pipeline = feature_pipeline
        self.calibration = calibration
        self.calibrator = None
        self.num_boost_round = 800
        self.early_stopping_rounds = 40

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        **kwargs
    ) -> 'CTRPredictorModel':
        """Train CTR model with class balancing."""
        # Set scale_pos_weight
        pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
        self.params['scale_pos_weight'] = pos_weight

        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)
            if X_val is not None:
                X_val = self.feature_pipeline.transform(X_val)

        dtrain = xgb.DMatrix(X, label=y)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, "train"), (dval, "val")]
        else:
            evals = [(dtrain, "train")]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=kwargs.get('verbose_eval', 50)
        )

        # Calibrate probabilities if requested
        if self.calibration and X_val is not None:
            from sklearn.isotonic import IsotonicRegression
            val_preds = self.model.predict(dval)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(val_preds, y_val)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated CTR."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)

        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)

        if self.calibration and self.calibrator:
            preds = self.calibrator.predict(preds)

        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)


class CVRPredictorModel(BaseModel):
    """Conversion rate prediction model (post-click)."""

    def __init__(
        self,
        model_spec: ModelSpec,
        params: Dict[str, Any] = None,
        feature_pipeline = None,
        calibration: bool = True
    ):
        super().__init__(model_spec)
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.03,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 20,
            "seed": 42,
            "nthread": -1,
            "verbosity": 1
        }
        self.feature_pipeline = feature_pipeline
        self.calibration = calibration
        self.calibrator = None
        self.num_boost_round = 600
        self.early_stopping_rounds = 30

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        **kwargs
    ) -> 'CVRPredictorModel':
        """Train CVR model (typically on clicked samples only)."""
        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)
            if X_val is not None:
                X_val = self.feature_pipeline.transform(X_val)

        dtrain = xgb.DMatrix(X, label=y)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, "train"), (dval, "val")]
        else:
            evals = [(dtrain, "train")]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=kwargs.get('verbose_eval', 50)
        )

        if self.calibration and X_val is not None:
            from sklearn.isotonic import IsotonicRegression
            val_preds = self.model.predict(dval)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(val_preds, y_val)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)

        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)

        if self.calibration and self.calibrator:
            preds = self.calibrator.predict(preds)

        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)


class AdRankerModel(BaseModel):
    """Ad ranking model with business constraints."""

    def __init__(
        self,
        model_spec: ModelSpec,
        params: Dict[str, Any] = None,
        feature_pipeline = None,
        bid_weight: float = 1.0,
        quality_weight: float = 1.0
    ):
        super().__init__(model_spec)
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "nthread": -1,
            "verbosity": 1
        }
        self.feature_pipeline = feature_pipeline
        self.bid_weight = bid_weight
        self.quality_weight = quality_weight
        self.num_boost_round = 1000
        self.early_stopping_rounds = 50

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        **kwargs
    ) -> 'AdRankerModel':
        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)
            if X_val is not None:
                X_val = self.feature_pipeline.transform(X_val)

        dtrain = xgb.DMatrix(X, label=y)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, "train"), (dval, "val")]
        else:
            evals = [(dtrain, "train")]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=kwargs.get('verbose_eval', 50)
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def rank_ads(
        self,
        user_features: Dict[str, Any],
        ad_candidates: List[Dict[str, Any]],
        bids: List[float]
    ) -> List[Dict[str, Any]]:
        """Rank ads using pCTR * bid (ECPM-based ranking)."""
        rows = []
        for i, ad in enumerate(ad_candidates):
            row = user_features.copy()
            row.update(ad)
            rows.append(row)

        X = pd.DataFrame(rows)
        pctr = self.predict_proba(X)

        # ECPM = pCTR * bid
        for i, ad in enumerate(ad_candidates):
            ad['pctr'] = float(pctr[i])
            ad['bid'] = bids[i] if i < len(bids) else 0.0
            ad['ecpm'] = ad['pctr'] * ad['bid'] * self.bid_weight

        # Sort by ECPM
        ad_candidates.sort(key=lambda x: x['ecpm'], reverse=True)
        return ad_candidates


class PersonalizationModel(BaseModel):
    """Personalization model for user-specific ranking."""

    def __init__(
        self,
        model_spec: ModelSpec,
        params: Dict[str, Any] = None,
        feature_pipeline = None,
        user_embedding_dim: int = 64
    ):
        super().__init__(model_spec)
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "nthread": -1,
            "verbosity": 1
        }
        self.feature_pipeline = feature_pipeline
        self.user_embedding_dim = user_embedding_dim
        self.num_boost_round = 800
        self.early_stopping_rounds = 40

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        **kwargs
    ) -> 'PersonalizationModel':
        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)
            if X_val is not None:
                X_val = self.feature_pipeline.transform(X_val)

        dtrain = xgb.DMatrix(X, label=y)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, "train"), (dval, "val")]
        else:
            evals = [(dtrain, "train")]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=kwargs.get('verbose_eval', 50)
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.feature_pipeline:
            X = self.feature_pipeline.transform(X)

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)

    def personalize_ranking(
        self,
        user_id: str,
        user_features: Dict[str, Any],
        items: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Personalize ranking for a specific user."""
        rows = []
        for item in items:
            row = user_features.copy()
            row.update(item)
            row['user_id'] = user_id
            rows.append(row)

        X = pd.DataFrame(rows)
        scores = self.predict_proba(X)

        for i, item in enumerate(items):
            item['personalized_score'] = float(scores[i])

        items.sort(key=lambda x: x['personalized_score'], reverse=True)
        return items[:top_k]


def create_model(model_spec: ModelSpec, feature_pipeline = None, **kwargs) -> BaseModel:
    """Factory function to create model by type."""
    model_type = model_spec.model_type

    if model_type == ModelType.RANKER:
        return RankingModel(model_spec, feature_pipeline=feature_pipeline, **kwargs)
    elif model_type == ModelType.RECOMMENDER:
        return RecommendationModel(model_spec, feature_pipeline=feature_pipeline, **kwargs)
    elif model_type == ModelType.SEARCH_RERANKER:
        return SearchRerankerModel(model_spec, feature_pipeline=feature_pipeline, **kwargs)
    elif model_type == ModelType.CTR_PREDICTOR:
        return CTRPredictorModel(model_spec, feature_pipeline=feature_pipeline, **kwargs)
    elif model_type == ModelType.CVR_PREDICTOR:
        return CVRPredictorModel(model_spec, feature_pipeline=feature_pipeline, **kwargs)
    elif model_type == ModelType.AD_RANKER:
        return AdRankerModel(model_spec, feature_pipeline=feature_pipeline, **kwargs)
    elif model_type == ModelType.PERSONALIZER:
        return PersonalizationModel(model_spec, feature_pipeline=feature_pipeline, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test model creation
    from core.schemas import ModelSpec, ModelType

    spec = ModelSpec(
        name="test_ranker",
        model_type=ModelType.RANKER,
        version="1.0.0"
    )

    model = create_model(spec)
    print(f"Created model: {type(model).__name__}")