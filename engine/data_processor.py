import sys
import pandas as pd
import numpy as np
from engine.logger import logger
from engine.exception import ThreatForecasterException
from engine.utils import reduce_mem_usage


class DataProcessor:
    """Handles feature engineering, dot-separated version parsing, date extraction, and type casting."""

    def __init__(self, config: dict):
        self.config = config
        self.version_cols = config["features"]["version_columns"]
        self.date_cols = config["features"]["date_columns"]

    def parse_versions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Splits version strings (e.g. 1.1.15100.1) into numeric components."""
        try:
            for col in self.version_cols:
                if col in df.columns:
                    ver_split = df[col].astype(str).str.split(".", expand=True)
                    for i in range(min(4, ver_split.shape[1])):
                        df[f"{col}_{i}"] = (
                            pd.to_numeric(ver_split[i], errors="coerce")
                            .fillna(-1)
                            .astype(np.float32)
                        )
                    df.drop(columns=[col], inplace=True)
            return df
        except Exception as e:
            raise ThreatForecasterException(e, sys)

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts numerical date parts from date features."""
        try:
            for col in self.date_cols:
                if col in df.columns:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = dt.dt.year.fillna(-1).astype(np.int16)
                    df[f"{col}_month"] = dt.dt.month.fillna(-1).astype(np.int8)
                    df[f"{col}_day"] = dt.dt.day.fillna(-1).astype(np.int8)
                    df[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(np.int8)
                    df.drop(columns=[col], inplace=True)
            return df
        except Exception as e:
            raise ThreatForecasterException(e, sys)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes full transformation pipeline."""
        try:
            logger.info("Starting feature engineering transformations...")
            df = df.copy()
            df = self.parse_versions(df)
            df = self.parse_dates(df)

            # Cast remaining text/object features to category dtype for LightGBM/CatBoost
            cat_cols = df.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                df[col] = df[col].astype("category")

            df = reduce_mem_usage(df)
            logger.info("Feature engineering completed successfully.")
            return df
        except Exception as e:
            raise ThreatForecasterException(e, sys)
