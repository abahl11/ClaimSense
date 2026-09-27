"""Learned text features from the claim description (Layers 2 and 3).

* Layer 2: TF-IDF on words and word pairs, compressed to a few dense components with
  truncated SVD. Both are fit inside the model pipeline, so on training rows only.
* Layer 3: sentence embeddings from a pretrained model (all-MiniLM-L6-v2), grouped with
  KMeans. The embedding model is used as-is, never trained on our data, so embedding a
  description cannot leak information; only the KMeans step learns, and it is fit on
  training rows inside the pipeline. Embeddings are cached on disk because they are slow
  to compute and identical every run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import Config

logger = logging.getLogger(__name__)

TEXT = "text"
CLUSTER_FEATURE = "embedding_cluster"


def tfidf_svd(cfg: Config) -> Pipeline:
    """Layer 2: TF-IDF (unigrams and bigrams) followed by truncated SVD."""
    t = cfg["tfidf"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(t["ngram_range"]),
                    min_df=t["min_df"],
                    sublinear_tf=t["sublinear_tf"],
                ),
            ),
            ("svd", TruncatedSVD(n_components=t["svd_components"], random_state=cfg.random_seed)),
        ]
    )


@lru_cache(maxsize=2)
def load_encoder(model_name: str) -> Any:
    """Load the sentence-transformer once per process (downloaded on first use)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer %s", model_name)
    return SentenceTransformer(model_name, device="cpu")


class EmbeddingStore:
    """Sentence embeddings keyed by description text, persisted to a Parquet file.

    Args:
        path: Cache file location.
        model_name: Sentence-transformer model to use for texts not yet cached.
        batch_size: Encoding batch size.
    """

    def __init__(self, path: Path, model_name: str, batch_size: int) -> None:
        self.path = path
        self.model_name = model_name
        self.batch_size = batch_size
        self._texts: dict[str, int] = {}
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        """Read the cache file if it exists and was built with the same model."""
        if not self.path.is_file():
            return
        table = pd.read_parquet(self.path)
        if table.attrs.get("model_name", self.model_name) != self.model_name:
            logger.warning("Ignoring embedding cache built with a different model")
            return
        self._texts = {text: i for i, text in enumerate(table[TEXT])}
        self._vectors = table.drop(columns=TEXT).to_numpy(dtype=np.float32)
        logger.info("Loaded %d cached embeddings from %s", len(self._texts), self.path)

    def _save(self) -> None:
        """Write every cached embedding to disk."""
        texts = sorted(self._texts, key=self._texts.get)
        table = pd.DataFrame(
            self._vectors, columns=[f"e{i}" for i in range(self._vectors.shape[1])]
        )
        table.insert(0, TEXT, texts)
        table.attrs["model_name"] = self.model_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(self.path, index=False)
        logger.info("Saved %d embeddings to %s", len(texts), self.path)

    def _add(self, texts: list[str]) -> None:
        """Encode texts that are not cached yet and store them."""
        logger.info("Encoding %d new descriptions with %s", len(texts), self.model_name)
        vectors = load_encoder(self.model_name).encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        start = len(self._texts)
        self._texts.update({text: start + i for i, text in enumerate(texts)})
        stacked = [self._vectors, vectors] if len(self._vectors) else [vectors]
        self._vectors = np.vstack(stacked).astype(np.float32)
        self._save()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one unit-length embedding row per text, computing any that are missing."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._texts]
        if missing:
            self._add(missing)
        return self._vectors[[self._texts[text] for text in texts]]


@lru_cache(maxsize=4)
def get_store(path: str, model_name: str, batch_size: int) -> EmbeddingStore:
    """Process-wide embedding store, so the cache is loaded once and never pickled."""
    return EmbeddingStore(Path(path), model_name, batch_size)


def embed_texts(texts: Sequence[str], path: str, model_name: str, batch_size: int) -> np.ndarray:
    """Embeddings for ``texts`` via the shared on-disk cache."""
    return get_store(path, model_name, batch_size).embed(list(texts))


def as_texts(X: Any) -> list[str]:
    """Flatten a column (Series, array or one-column frame) into a list of strings."""
    values = np.asarray(X, dtype=object).ravel()
    return ["" if pd.isna(value) else str(value) for value in values]


class EmbeddingClusters(BaseEstimator, TransformerMixin):
    """Layer 3: assign each description to a KMeans cluster of its sentence embedding.

    Only the cache location and model name are stored on the estimator, so saving the
    fitted pipeline does not pickle the (large) embedding model.

    Args:
        n_clusters: Number of KMeans clusters.
        cache_path: Embedding cache file.
        model_name: Sentence-transformer model name.
        batch_size: Encoding batch size.
        n_init: KMeans restarts.
        random_state: Seed for KMeans.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        cache_path: str = "",
        model_name: str = "",
        batch_size: int = 256,
        n_init: int = 10,
        random_state: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.cache_path = cache_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_init = n_init
        self.random_state = random_state

    def _embed(self, X: Any) -> np.ndarray:
        return embed_texts(as_texts(X), self.cache_path, self.model_name, self.batch_size)

    def fit(self, X: Any, y: object = None) -> EmbeddingClusters:
        """Fit KMeans on the embeddings of the (training) descriptions."""
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state
        ).fit(self._embed(X))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Nearest-cluster id for each description, as a single column."""
        return self.kmeans_.predict(self._embed(X)).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Name of the single output column."""
        return np.array([CLUSTER_FEATURE], dtype=object)


def embedding_clusters(cfg: Config) -> Pipeline:
    """Layer 3 pipeline: embedding cluster id, one-hot encoded."""
    e = cfg["embeddings"]
    if e["n_clusters"] is None:
        raise ValueError("embeddings.n_clusters is not set; run --stage features and choose k.")
    clusters = EmbeddingClusters(
        n_clusters=e["n_clusters"],
        cache_path=str(cfg.path("embeddings_cache")),
        model_name=e["model_name"],
        batch_size=e["batch_size"],
        n_init=e["kmeans_n_init"],
        random_state=cfg.random_seed,
    )
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return Pipeline([("clusters", clusters), ("onehot", onehot)])


def name_clusters(kmeans: KMeans, cfg: Config) -> dict[int, str]:
    """Map cluster ids to names via the anchor descriptions in the config.

    Each configured name belongs to whichever cluster its anchor description falls in, so
    names stay correct even if cluster numbering changes. Clusters without an anchor are
    labelled "unnamed"; two anchors landing in one cluster are joined and logged.
    """
    anchors = cfg["embeddings"]["cluster_names"]
    names: dict[int, str] = {}
    if anchors:
        ids = kmeans.predict(embeddings_for(pd.Series(list(anchors.values())), cfg))
        for name, cluster in zip(anchors, ids, strict=True):
            if int(cluster) in names:
                logger.warning(
                    "Anchors for '%s' and '%s' share a cluster", names[int(cluster)], name
                )
                names[int(cluster)] += f" / {name}"
            else:
                names[int(cluster)] = name
    return {c: names.get(c, f"cluster {c} (unnamed)") for c in range(kmeans.n_clusters)}


def embeddings_for(texts: pd.Series, cfg: Config) -> np.ndarray:
    """Embeddings for a column of descriptions using the configured model and cache."""
    e = cfg["embeddings"]
    return embed_texts(
        as_texts(texts), str(cfg.path("embeddings_cache")), e["model_name"], e["batch_size"]
    )
