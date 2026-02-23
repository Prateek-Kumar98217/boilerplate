"""
End-to-end ML data preprocessing example.

Run:
    python examples/ml_pipeline.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from ml.analysis import DataAnalyzer
from ml.cleaning import DataCleaner, stratified_split
from ml.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    # ── 1. Generate synthetic dataset ─────────────────────────────────
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(20)])
    df["label"] = y

    # Introduce synthetic noise
    rng = np.random.default_rng(42)
    df.loc[rng.choice(1000, 50, replace=False), "feat_0"] = np.nan
    df.loc[rng.choice(1000, 20, replace=False), "feat_1"] = 999.0

    print(f"Dataset shape: {df.shape}")

    # ── 2. EDA ─────────────────────────────────────────────────────────
    analyzer = DataAnalyzer(df, target_col="label")
    missing = analyzer.missing_summary()
    print("\nMissing values:\n", missing)
    print("\nClass balance:\n", analyzer.class_balance())
    outliers = analyzer.outlier_summary()
    print("\nOutlier summary (top 5):\n", outliers.head())

    # ── 3. Train / val / test split ────────────────────────────────────
    df_train, df_val, df_test = stratified_split(
        df, "label", test_size=0.2, val_size=0.1
    )

    # ── 4. Cleaning ────────────────────────────────────────────────────
    cleaner = DataCleaner(
        missing_strategy="median", outlier_method="iqr", target_col="label"
    )
    df_train_clean = cleaner.fit_transform(df_train)
    df_val_clean = cleaner.transform(df_val)
    df_test_clean = cleaner.transform(df_test)
    print(
        f"\nAfter cleaning — train: {df_train_clean.shape}, val: {df_val_clean.shape}"
    )

    # ── 5. Feature engineering ─────────────────────────────────────────
    numeric_cols = [c for c in df_train_clean.columns if c != "label"]
    fe = FeatureEngineer(scaler="standard", pca_components=10)
    X_train = fe.fit_transform(
        df_train_clean, numeric_cols, target=df_train_clean["label"]
    )
    X_val = fe.transform(df_val_clean)
    X_test = fe.transform(df_test_clean)
    print(f"Feature matrix shapes — train: {X_train.shape}, val: {X_val.shape}")

    if fe.pca_explained_variance_ratio is not None:
        print(
            f"PCA cumulative variance (10 components): "
            f"{fe.pca_explained_variance_ratio.sum():.3f}"
        )


if __name__ == "__main__":
    main()
