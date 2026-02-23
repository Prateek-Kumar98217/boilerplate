"""
Feature engineering — scaling, encoding, selection, dimensionality reduction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


class FeatureEngineer:
    """
    Sklearn-compatible feature engineering pipeline.

    Example:
        fe = FeatureEngineer(
            scaler="standard",
            categorical_encoding="onehot",
            pca_components=50,
        )
        X_train = fe.fit_transform(df_train, numeric_cols, cat_cols)
        X_val   = fe.transform(df_val, numeric_cols, cat_cols)
    """

    def __init__(
        self,
        scaler: str = "standard",  # standard | minmax | robust | none
        categorical_encoding: str = "onehot",  # onehot | ordinal | target | none
        polynomial_degree: int = 1,
        pca_components: Optional[int] = None,
        select_k_best: Optional[int] = None,
        task: str = "classification",  # classification | regression
    ) -> None:
        self.scaler_type = scaler
        self.cat_encoding = categorical_encoding
        self.poly_degree = polynomial_degree
        self.pca_components = pca_components
        self.select_k_best = select_k_best
        self.task = task

        # Fitted objects
        self._scaler = None
        self._encoder: Optional[OneHotEncoder] = None
        self._poly: Optional[PolynomialFeatures] = None
        self._pca = None
        self._selector = None
        self._numeric_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._fitted = False

    def _init_scaler(self):
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        return None

    def fit(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        cat_cols: Optional[List[str]] = None,
        target: Optional[pd.Series] = None,
    ) -> "FeatureEngineer":
        self._numeric_cols = numeric_cols
        self._cat_cols = cat_cols or []
        X_num = df[numeric_cols].values.astype(float)

        # Polynomial features
        if self.poly_degree > 1:
            self._poly = PolynomialFeatures(self.poly_degree, include_bias=False)
            X_num = self._poly.fit_transform(X_num)

        # Scaling
        self._scaler = self._init_scaler()
        if self._scaler:
            self._scaler.fit(X_num)
            X_num = self._scaler.transform(X_num)

        # Categorical encoding
        if self._cat_cols:
            if self.cat_encoding == "onehot":
                self._encoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )
                self._encoder.fit(df[self._cat_cols].astype(str))
            elif self.cat_encoding == "target" and target is not None:
                self._encoder = TargetEncoder()
                self._encoder.fit(df[self._cat_cols], target)

        # Combine numeric + categorical for downstream steps
        X = self._apply_encoding(df, X_num)

        # Feature selection
        if self.select_k_best and target is not None:
            score_fn = f_classif if self.task == "classification" else f_regression
            self._selector = SelectKBest(
                score_fn, k=min(self.select_k_best, X.shape[1])
            )
            self._selector.fit(X, target.values)

        # PCA
        if self.pca_components and X.shape[1] > self.pca_components:
            self._pca = PCA(n_components=self.pca_components, random_state=42)
            X_sel = self._selector.transform(X) if self._selector else X
            self._pca.fit(X_sel)

        self._fitted = True
        return self

    def _apply_encoding(self, df: pd.DataFrame, X_num: np.ndarray) -> np.ndarray:
        if not self._cat_cols or self._encoder is None:
            return X_num
        X_cat = self._encoder.transform(df[self._cat_cols].astype(str))
        return np.hstack([X_num, X_cat])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Call fit() first."
        X_num = df[self._numeric_cols].values.astype(float)
        if self._poly:
            X_num = self._poly.transform(X_num)
        if self._scaler:
            X_num = self._scaler.transform(X_num)
        X = self._apply_encoding(df, X_num)
        if self._selector:
            X = self._selector.transform(X)
        if self._pca:
            X = self._pca.transform(X)
        return X

    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        cat_cols: Optional[List[str]] = None,
        target: Optional[pd.Series] = None,
    ) -> np.ndarray:
        return self.fit(df, numeric_cols, cat_cols, target).transform(df)

    @property
    def pca_explained_variance_ratio(self) -> Optional[np.ndarray]:
        return self._pca.explained_variance_ratio_ if self._pca else None
