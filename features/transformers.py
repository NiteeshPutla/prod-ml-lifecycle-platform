"""Feature transformers for preprocessing and feature engineering."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """Base class for feature transformers."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureTransformer':
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureTransformer':
        with open(path, 'rb') as f:
            return pickle.load(f)

    # sklearn compatibility
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for sklearn compatibility."""
        # Exclude fitted attributes (those not in __init__ signature)
        import inspect
        init_params = inspect.signature(self.__init__).parameters.keys()
        return {k: v for k, v in self.__dict__.items() if k in init_params}

    def set_params(self, **params) -> 'FeatureTransformer':
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __sklearn_tags__(self):
        """Return sklearn tags for compatibility."""
        from sklearn.utils import Tags
        return Tags()


class NumericalTransformer(FeatureTransformer):
    """Transformer for numerical features."""

    def __init__(
        self,
        method: str = "standard",  # standard, minmax, robust, log, sqrt, boxcox, yeo-johnson
        clip_outliers: bool = True,
        outlier_std: float = 3.0,
        fill_na: str = "median"  # median, mean, zero, drop
    ):
        self.method = method
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        self.fill_na = fill_na
        self.scaler = None
        self.fill_value = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'NumericalTransformer':
        self.feature_names = X.columns.tolist()

        # Compute fill value
        if self.fill_na == "median":
            self.fill_value = X.median()
        elif self.fill_na == "mean":
            self.fill_value = X.mean()
        elif self.fill_na == "zero":
            self.fill_value = 0
        else:
            self.fill_value = X.median()

        X_filled = X.fillna(self.fill_value)

        # Clip outliers
        if self.clip_outliers:
            for col in X_filled.columns:
                mean, std = X_filled[col].mean(), X_filled[col].std()
                if std > 0:
                    lower = mean - self.outlier_std * std
                    upper = mean + self.outlier_std * std
                    X_filled[col] = X_filled[col].clip(lower, upper)

        # Fit scaler
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method in ["log", "sqrt", "boxcox", "yeo-johnson"]:
            self.scaler = None  # Handled in transform
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.scaler:
            self.scaler.fit(X_filled)

        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names for sklearn compatibility."""
        if input_features is None:
            input_features = self.feature_names
        return np.array(input_features) if input_features is not None else np.array([])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            raise ValueError("Transformer not fitted")

        X = X[self.feature_names].copy()
        X = X.fillna(self.fill_value)

        if self.method == "log":
            # Log transform with offset for zeros/negatives
            for col in X.columns:
                min_val = X[col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                X[col] = np.log1p(X[col] + offset)
        elif self.method == "sqrt":
            for col in X.columns:
                min_val = X[col].min()
                offset = abs(min_val) + 1 if min_val < 0 else 0
                X[col] = np.sqrt(X[col] + offset)
        elif self.method == "boxcox":
            from scipy.stats import boxcox
            for col in X.columns:
                min_val = X[col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                X[col], _ = boxcox(X[col] + offset)
        elif self.method == "yeo-johnson":
            from scipy.stats import yeojohnson
            for col in X.columns:
                X[col], _ = yeojohnson(X[col])
        elif self.scaler:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return X


class CategoricalTransformer(FeatureTransformer):
    """Transformer for categorical features."""

    def __init__(
        self,
        method: str = "label",  # label, onehot, target, frequency, embedding
        max_categories: int = 100,
        min_frequency: int = 10,
        handle_unknown: str = "ignore",
        embedding_dim: int = None
    ):
        self.method = method
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        self.embedding_dim = embedding_dim
        self.encoders = {}
        self.category_maps = {}
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CategoricalTransformer':
        self.feature_names = X.columns.tolist()

        for col in X.columns:
            series = X[col].astype(str).fillna("__missing__")

            # Compute frequency
            freq = series.value_counts()
            frequent_cats = freq[freq >= self.min_frequency].index.tolist()

            if len(frequent_cats) > self.max_categories:
                frequent_cats = frequent_cats[:self.max_categories]

            self.category_maps[col] = {cat: i for i, cat in enumerate(frequent_cats)}
            self.category_maps[col]["__unknown__"] = len(frequent_cats)

            if self.method == "label":
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(frequent_cats + ["__unknown__"])
            elif self.method == "onehot":
                self.encoders[col] = OneHotEncoder(
                    categories=[frequent_cats + ["__unknown__"]],
                    handle_unknown=self.handle_unknown,
                    sparse_output=False
                )
                self.encoders[col].fit(series.to_frame())
            elif self.method == "target" and y is not None:
                # Target encoding
                target_mean = y.groupby(series).mean()
                global_mean = y.mean()
                self.encoders[col] = target_mean.to_dict()
                self.encoders[col]["__global__"] = global_mean
            elif self.method == "frequency":
                self.encoders[col] = (freq / len(series)).to_dict()
                self.encoders[col]["__unknown__"] = 0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            raise ValueError("Transformer not fitted")

        result = pd.DataFrame(index=X.index)

        for col in self.feature_names:
            if col not in X.columns:
                continue

            series = X[col].astype(str).fillna("__missing__")

            if self.method == "label":
                # Map unknown to __unknown__ - category_maps already provides integer encoding
                mapped = series.map(self.category_maps[col]).fillna(self.category_maps[col]["__unknown__"])
                result[f"{col}_encoded"] = mapped.astype(int)

            elif self.method == "onehot":
                encoded = self.encoders[col].transform(series.to_frame())
                feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                result = pd.concat([result, encoded_df], axis=1)

            elif self.method == "target":
                mapped = series.map(self.encoders[col]).fillna(self.encoders[col]["__global__"])
                result[f"{col}_target_enc"] = mapped

            elif self.method == "frequency":
                mapped = series.map(self.encoders[col]).fillna(0)
                result[f"{col}_freq_enc"] = mapped

            elif self.method == "embedding":
                # Return indices for embedding lookup
                mapped = series.map(self.category_maps[col]).fillna(self.category_maps[col]["__unknown__"])
                result[f"{col}_idx"] = mapped.astype(int)

        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names for sklearn compatibility."""
        if self.feature_names is None:
            return np.array([])

        if self.method == "label" or self.method == "target" or self.method == "frequency" or self.method == "embedding":
            return np.array([f"{col}_encoded" if self.method == "label"
                            else f"{col}_target_enc" if self.method == "target"
                            else f"{col}_freq_enc" if self.method == "frequency"
                            else f"{col}_idx" for col in self.feature_names])
        elif self.method == "onehot":
            # For onehot, we need to get the categories from the encoder
            names = []
            for col in self.feature_names:
                if col in self.encoders:
                    cats = self.encoders[col].categories_[0]
                    names.extend([f"{col}_{cat}" for cat in cats])
            return np.array(names)
        return np.array(input_features) if input_features is not None else np.array([])


class TextTransformer(FeatureTransformer):
    """Transformer for text features."""

    def __init__(
        self,
        method: str = "tfidf",  # tfidf, count, hash
        max_features: int = 1000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'TextTransformer':
        self.feature_names = X.columns.tolist()

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df
            )
        elif self.method == "count":
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df
            )
        elif self.method == "hash":
            from sklearn.feature_extraction.text import HashingVectorizer
            self.vectorizer = HashingVectorizer(
                n_features=self.max_features,
                ngram_range=self.ngram_range
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Combine all text columns
        if len(X.columns) == 1:
            texts = X.iloc[:, 0].astype(str).fillna("")
        else:
            texts = X.astype(str).fillna("").agg(" ".join, axis=1)

        self.vectorizer.fit(texts)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            raise ValueError("Transformer not fitted")

        if len(X.columns) == 1:
            texts = X.iloc[:, 0].astype(str).fillna("")
        else:
            texts = X.astype(str).fillna("").agg(" ".join, axis=1)

        vectors = self.vectorizer.transform(texts)

        if hasattr(vectors, 'toarray'):
            vectors = vectors.toarray()

        feature_names = [f"text_{i}" for i in range(vectors.shape[1])]
        return pd.DataFrame(vectors, columns=feature_names, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names for sklearn compatibility."""
        if self.vectorizer is None:
            return np.array([])
        return np.array([f"text_{i}" for i in range(self.vectorizer.get_feature_names_out().shape[0])])


class EmbeddingTransformer(FeatureTransformer):
    """Transformer for pre-trained embeddings."""

    def __init__(
        self,
        embedding_dim: int = 64,
        aggregation: str = "mean"  # mean, sum, max, concat
    ):
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'EmbeddingTransformer':
        self.feature_names = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            raise ValueError("Transformer not fitted")

        result = pd.DataFrame(index=X.index)

        for col in self.feature_names:
            if col not in X.columns:
                continue

            # X[col] should contain lists/arrays of embeddings
            embeddings = X[col].apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.zeros(self.embedding_dim))

            if self.aggregation == "mean":
                agg = embeddings.apply(lambda x: x.mean(axis=0) if len(x) > 0 else np.zeros(self.embedding_dim))
            elif self.aggregation == "sum":
                agg = embeddings.apply(lambda x: x.sum(axis=0) if len(x) > 0 else np.zeros(self.embedding_dim))
            elif self.aggregation == "max":
                agg = embeddings.apply(lambda x: x.max(axis=0) if len(x) > 0 else np.zeros(self.embedding_dim))
            elif self.aggregation == "concat":
                # Pad or truncate to fixed length
                max_seq = 10
                agg = embeddings.apply(lambda x: np.pad(x[:max_seq], ((0, max_seq - len(x)), (0, 0))).flatten() if len(x) > 0 else np.zeros(self.embedding_dim * max_seq))
            else:
                agg = embeddings.apply(lambda x: x.mean(axis=0) if len(x) > 0 else np.zeros(self.embedding_dim))

            # Expand to columns
            for i in range(len(agg.iloc[0])):
                result[f"{col}_emb_{i}"] = agg.apply(lambda x: x[i])

        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names for sklearn compatibility."""
        if self.feature_names is None:
            return np.array([])
        names = []
        for col in self.feature_names:
            for i in range(self.embedding_dim):
                names.append(f"{col}_emb_{i}")
        return np.array(names)


class CrossFeatureTransformer(FeatureTransformer):
    """Generate cross/interaction features."""

    def __init__(
        self,
        feature_pairs: List[tuple] = None,
        operations: List[str] = None,  # multiply, divide, add, subtract
        max_interactions: int = 50
    ):
        self.feature_pairs = feature_pairs
        self.operations = operations or ["multiply"]
        self.max_interactions = max_interactions
        self.feature_names = None
        self.selected_pairs = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CrossFeatureTransformer':
        self.feature_names = X.columns.tolist()

        if self.feature_pairs is None:
            # Auto-generate pairs from numerical features
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            pairs = []
            for i, c1 in enumerate(numerical_cols):
                for c2 in numerical_cols[i+1:]:
                    pairs.append((c1, c2))
                    if len(pairs) >= self.max_interactions:
                        break
                if len(pairs) >= self.max_interactions:
                    break
            self.feature_pairs = pairs

        self.selected_pairs = self.feature_pairs[:self.max_interactions]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_pairs is None:
            raise ValueError("Transformer not fitted")

        result = X.copy()

        for c1, c2 in self.selected_pairs:
            if c1 not in X.columns or c2 not in X.columns:
                continue

            for op in self.operations:
                if op == "multiply":
                    result[f"{c1}_x_{c2}"] = X[c1] * X[c2]
                elif op == "divide":
                    result[f"{c1}_div_{c2}"] = X[c1] / (X[c2] + 1e-8)
                elif op == "add":
                    result[f"{c1}_add_{c2}"] = X[c1] + X[c2]
                elif op == "subtract":
                    result[f"{c1}_sub_{c2}"] = X[c1] - X[c2]

        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return output feature names for sklearn compatibility."""
        if self.selected_pairs is None:
            return np.array(input_features) if input_features is not None else np.array([])
        names = list(input_features) if input_features is not None else []
        for c1, c2 in self.selected_pairs:
            for op in self.operations:
                if op == "multiply":
                    names.append(f"{c1}_x_{c2}")
                elif op == "divide":
                    names.append(f"{c1}_div_{c2}")
                elif op == "add":
                    names.append(f"{c1}_add_{c2}")
                elif op == "subtract":
                    names.append(f"{c1}_sub_{c2}")
        return np.array(names)


class FeaturePipeline:
    """Complete feature engineering pipeline."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.transformers = {}
        self.column_transformer = None
        self.is_fitted = False
        self.input_features = None
        self.output_features = None

    def build(self, feature_specs: Dict[str, Dict[str, Any]]) -> 'FeaturePipeline':
        """Build pipeline from feature specifications.

        feature_specs = {
            "user_age": {"type": "numerical", "method": "standard"},
            "user_gender": {"type": "categorical", "method": "label"},
            "item_description": {"type": "text", "method": "tfidf"},
            "user_embedding": {"type": "embedding", "embedding_dim": 64},
        }
        """
        transformers_list = []

        for feature_name, spec in feature_specs.items():
            ftype = spec.get("type", "numerical")

            if ftype == "numerical":
                transformer = NumericalTransformer(
                    method=spec.get("method", "standard"),
                    clip_outliers=spec.get("clip_outliers", True),
                    fill_na=spec.get("fill_na", "median")
                )
            elif ftype == "categorical":
                transformer = CategoricalTransformer(
                    method=spec.get("method", "label"),
                    max_categories=spec.get("max_categories", 100),
                    min_frequency=spec.get("min_frequency", 10)
                )
            elif ftype == "text":
                transformer = TextTransformer(
                    method=spec.get("method", "tfidf"),
                    max_features=spec.get("max_features", 1000)
                )
            elif ftype == "embedding":
                transformer = EmbeddingTransformer(
                    embedding_dim=spec.get("embedding_dim", 64),
                    aggregation=spec.get("aggregation", "mean")
                )
            else:
                logger.warning(f"Unknown feature type: {ftype} for {feature_name}")
                continue

            self.transformers[feature_name] = transformer
            transformers_list.append((feature_name, transformer, [feature_name]))

        # Add cross features if specified
        if self.config.get("cross_features", False):
            cross_transformer = CrossFeatureTransformer(
                max_interactions=self.config.get("max_interactions", 50)
            )
            self.transformers["cross_features"] = cross_transformer
            transformers_list.append(("cross_features", cross_transformer, "passthrough"))

        self.column_transformer = ColumnTransformer(
            transformers=transformers_list,
            remainder="drop",
            verbose_feature_names_out=False
        )

        return self

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeaturePipeline':
        """Fit the entire pipeline."""
        self.input_features = X.columns.tolist()
        self.column_transformer.fit(X, y)
        self.is_fitted = True

        # Get output feature names
        self.output_features = self.column_transformer.get_feature_names_out().tolist()
        logger.info(f"Pipeline fitted: {len(self.input_features)} input -> {len(self.output_features)} output features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted")

        X_transformed = self.column_transformer.transform(X)

        # ColumnTransformer returns numpy array
        return pd.DataFrame(X_transformed, columns=self.output_features, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Save entire pipeline."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FeaturePipeline':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_feature_importance_names(self) -> List[str]:
        """Get output feature names for model feature importance."""
        return self.output_features or []


def create_ranking_feature_pipeline() -> FeaturePipeline:
    """Create a standard feature pipeline for ranking models."""

    feature_specs = {
        # User features
        "user_age": {"type": "numerical", "method": "standard"},
        "user_gender": {"type": "categorical", "method": "label"},
        "user_country": {"type": "categorical", "method": "label"},
        "user_past_ctr": {"type": "numerical", "method": "log"},
        "user_session_count_7d": {"type": "numerical", "method": "sqrt"},
        "user_embedding": {"type": "embedding", "embedding_dim": 64},

        # Item features
        "item_category": {"type": "categorical", "method": "label"},
        "item_price": {"type": "numerical", "method": "log"},
        "item_ctr_7d": {"type": "numerical", "method": "log"},
        "item_cvr_7d": {"type": "numerical", "method": "log"},
        "item_embedding": {"type": "embedding", "embedding_dim": 64},

        # Cross features
        "position": {"type": "numerical", "method": "standard"},
        "user_item_affinity": {"type": "numerical", "method": "standard"},
    }

    config = {
        "cross_features": True,
        "max_interactions": 30
    }

    pipeline = FeaturePipeline(config)
    pipeline.build(feature_specs)
    return pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test pipeline
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "user_age": np.random.randint(18, 70, n),
        "user_gender": np.random.choice(["M", "F", "O"], n),
        "user_country": np.random.choice(["US", "UK", "CA", "DE"], n),
        "user_past_ctr": np.random.beta(1, 100, n),
        "user_session_count_7d": np.random.poisson(5, n),
        "user_embedding": [np.random.randn(64).tolist() for _ in range(n)],
        "item_category": np.random.choice(["electronics", "clothing", "books"], n),
        "item_price": np.random.lognormal(3, 1, n),
        "item_ctr_7d": np.random.beta(2, 100, n),
        "item_cvr_7d": np.random.beta(1, 200, n),
        "item_embedding": [np.random.randn(64).tolist() for _ in range(n)],
        "position": np.random.randint(1, 11, n),
        "user_item_affinity": np.random.random(n),
    })

    y = np.random.randint(0, 2, n)

    pipeline = create_ranking_feature_pipeline()
    X_transformed = pipeline.fit_transform(df, y)

    print(f"Input shape: {df.shape}")
    print(f"Output shape: {X_transformed.shape}")
    print(f"Output features: {X_transformed.columns.tolist()[:20]}...")