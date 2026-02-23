"""
ML dataset analysis utilities — EDA in code.

Covers:
- Descriptive statistics
- Distribution analysis (skewness, kurtosis)
- Correlation matrix
- Class imbalance detection
- Missing value analysis
- Outlier summary
- Feature importance via mutual information
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class DataAnalyzer:
    """
    EDA helper for tabular data.

    Example:
        df = pd.read_csv("data.csv")
        analyzer = DataAnalyzer(df, target_col="label")
        print(analyzer.basic_stats())
        print(analyzer.missing_summary())
        print(analyzer.class_balance())
        print(analyzer.feature_importance())
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        self.df = df.copy()
        self.target_col = target_col
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # ── Descriptive stats ─────────────────────────────────────────────

    def basic_stats(self) -> pd.DataFrame:
        """Extended describe: adds skewness, kurtosis, coefficient of variation."""
        desc = self.df[self._numeric_cols].describe()
        desc.loc["skewness"] = self.df[self._numeric_cols].skew()
        desc.loc["kurtosis"] = self.df[self._numeric_cols].kurtosis()
        cv = desc.loc["std"] / (desc.loc["mean"].replace(0, np.nan))
        desc.loc["cv"] = cv
        return desc

    # ── Missing values ────────────────────────────────────────────────

    def missing_summary(self) -> pd.DataFrame:
        missing = self.df.isnull().sum()
        pct = 100 * missing / len(self.df)
        return (
            pd.DataFrame({"missing_count": missing, "missing_pct": pct})
            .query("missing_count > 0")
            .sort_values("missing_pct", ascending=False)
        )

    # ── Outlier summary ───────────────────────────────────────────────

    def outlier_summary(
        self, method: str = "iqr", threshold: float = 3.0
    ) -> pd.DataFrame:
        results = []
        for col in self._numeric_cols:
            series = self.df[col].dropna()
            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                n_out = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
            else:  # z-score
                z = np.abs(stats.zscore(series))
                n_out = (z > threshold).sum()
            results.append(
                {"column": col, "n_outliers": n_out, "pct": 100 * n_out / len(series)}
            )
        return pd.DataFrame(results).sort_values("pct", ascending=False)

    # ── Correlation ───────────────────────────────────────────────────

    def correlation_matrix(self, method: str = "pearson") -> pd.DataFrame:
        return self.df[self._numeric_cols].corr(method=method)

    def high_correlations(
        self, threshold: float = 0.85
    ) -> List[Tuple[str, str, float]]:
        corr = self.correlation_matrix().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = []
        for col in upper.columns:
            correlated = upper[col][upper[col] > threshold]
            for row, val in correlated.items():
                pairs.append((row, col, round(val, 4)))
        return sorted(pairs, key=lambda x: -x[2])

    # ── Class balance ─────────────────────────────────────────────────

    def class_balance(self) -> Optional[pd.DataFrame]:
        if not self.target_col or self.target_col not in self.df.columns:
            return None
        counts = self.df[self.target_col].value_counts()
        pct = 100 * counts / counts.sum()
        return pd.DataFrame({"count": counts, "pct": pct})

    # ── Feature importance ────────────────────────────────────────────

    def feature_importance(self, task: str = "classification") -> pd.DataFrame:
        """
        Compute mutual information between each feature and the target.
        task: 'classification' | 'regression'
        """
        if not self.target_col:
            raise ValueError("target_col must be set to compute feature importance.")
        feature_cols = [c for c in self._numeric_cols if c != self.target_col]
        X = self.df[feature_cols].fillna(0).values
        y = self.df[self.target_col].values

        fn = mutual_info_classif if task == "classification" else mutual_info_regression
        scores = fn(X, y, random_state=42)
        return pd.DataFrame(
            {"feature": feature_cols, "mutual_info": scores}
        ).sort_values("mutual_info", ascending=False)

    # ── Full report ───────────────────────────────────────────────────

    def report(self) -> Dict[str, object]:
        return {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.to_dict(),
            "basic_stats": self.basic_stats(),
            "missing": self.missing_summary(),
            "outliers": self.outlier_summary(),
            "high_correlations": self.high_correlations(),
            "class_balance": self.class_balance(),
        }
