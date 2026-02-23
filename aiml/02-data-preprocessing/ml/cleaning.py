"""
ML data cleaning — missing values, outliers, deduplication, type coercion.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Production data cleaning pipeline for tabular ML datasets.

    Example:
        cleaner = DataCleaner(missing_strategy="median", outlier_method="iqr")
        df_clean = cleaner.fit_transform(df_train)
        df_test_clean = cleaner.transform(df_test)
    """

    def __init__(
        self,
        missing_strategy: str = "median",  # median | mean | mode | knn | drop
        outlier_method: str = "iqr",  # iqr | zscore | isolation_forest | none
        outlier_threshold: float = 3.0,
        remove_duplicates: bool = True,
        target_col: Optional[str] = None,
    ) -> None:
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.remove_duplicates = remove_duplicates
        self.target_col = target_col

        self._numeric_imputer = None
        self._cat_encoder: Dict[str, LabelEncoder] = {}
        self._numeric_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._iso_forest = None
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        df = df.copy()
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if self.target_col and self.target_col in self._numeric_cols:
            self._numeric_cols.remove(self.target_col)
        if self.target_col and self.target_col in self._cat_cols:
            self._cat_cols.remove(self.target_col)

        # Fit numeric imputer
        if self.missing_strategy == "knn":
            self._numeric_imputer = KNNImputer(n_neighbors=5)
        elif self.missing_strategy in ("median", "mean", "most_frequent"):
            strategy = (
                "most_frequent"
                if self.missing_strategy == "mode"
                else self.missing_strategy
            )
            self._numeric_imputer = SimpleImputer(strategy=strategy)
        if self._numeric_imputer and len(self._numeric_cols) > 0:
            self._numeric_imputer.fit(df[self._numeric_cols])

        # Fit categorical encoders
        for col in self._cat_cols:
            le = LabelEncoder()
            le.fit(df[col].fillna("_MISSING_").astype(str))
            self._cat_encoder[col] = le

        # Fit isolation forest
        if self.outlier_method == "isolation_forest" and len(self._numeric_cols) > 0:
            self._iso_forest = IsolationForest(contamination=0.05, random_state=42)
            self._iso_forest.fit(df[self._numeric_cols].fillna(0))

        self._fitted = True
        return self

    # ── Transform ────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Call fit() before transform()"
        df = df.copy()

        # Dedup
        if self.remove_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            logger.info("Removed %d duplicate rows", before - len(df))

        # Impute numeric
        if self._numeric_imputer and len(self._numeric_cols) > 0:
            missing_mask = df[self._numeric_cols].isnull().any(axis=1)
            if missing_mask.any():
                df.loc[:, self._numeric_cols] = self._numeric_imputer.transform(
                    df[self._numeric_cols]
                )

        # Drop rows if strategy == "drop"
        if self.missing_strategy == "drop":
            before = len(df)
            df = df.dropna(subset=self._numeric_cols)
            logger.info("Dropped %d rows with missing values", before - len(df))

        # Encode categoricals
        for col, le in self._cat_encoder.items():
            if col in df.columns:
                series = df[col].fillna("_MISSING_").astype(str)
                # Handle unseen labels gracefully
                known = set(le.classes_)
                series = series.map(lambda x: x if x in known else "_MISSING_")
                df[col] = le.transform(series)

        # Remove outliers
        if self.outlier_method == "iqr" and len(self._numeric_cols) > 0:
            for col in self._numeric_cols:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        elif self.outlier_method == "zscore" and len(self._numeric_cols) > 0:
            for col in self._numeric_cols:
                z = np.abs(stats.zscore(df[col].fillna(df[col].median())))
                df.loc[z > self.outlier_threshold, col] = df[col].median()

        elif self.outlier_method == "isolation_forest" and self._iso_forest is not None:
            preds = self._iso_forest.predict(df[self._numeric_cols].fillna(0))
            n_removed = (preds == -1).sum()
            df = df[preds == 1].reset_index(drop=True)
            logger.info("Isolation Forest removed %d outlier rows", n_removed)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ── Train / Val / Test split ──────────────────────────────────────────


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split.

    Returns:
        (df_train, df_val, df_test)
    """
    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_train,
        test_size=relative_val,
        stratify=df_train[target_col],
        random_state=random_state,
    )
    logger.info(
        "Split → train=%d val=%d test=%d", len(df_train), len(df_val), len(df_test)
    )
    return df_train, df_val, df_test
