import sys
import pandas as pd
import numpy as np
from engine.logger import logger
from engine.exception import ThreatForecasterException
from engine.utils import reduce_mem_usage


class DataProcessor:
    """Handles feature engineering, version parsing, hardware ratios, and frequency encoding."""

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

    def add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers system hardware ratios and capacity metrics."""
        try:
            # 1. RAM per Core Ratio
            if (
                "TotalPhysicalRAMMB" in df.columns
                and "ProcessorCoreCount" in df.columns
            ):
                df["RAM_per_Core"] = (
                    df["TotalPhysicalRAMMB"] / (df["ProcessorCoreCount"] + 1e-5)
                ).astype(np.float32)

            # 2. Disk Storage Utilization Ratio
            if (
                "SystemVolumeCapacityMB" in df.columns
                and "PrimaryDiskCapacityMB" in df.columns
            ):
                df["Disk_Space_Ratio"] = (
                    df["SystemVolumeCapacityMB"] / (df["PrimaryDiskCapacityMB"] + 1e-5)
                ).astype(np.float32)

            # 3. Screen Resolution Pixel Count
            if (
                "PrimaryDisplayResolutionHorizontal" in df.columns
                and "PrimaryDisplayResolutionVertical" in df.columns
            ):
                df["Screen_Pixel_Area"] = (
                    df["PrimaryDisplayResolutionHorizontal"]
                    * df["PrimaryDisplayResolutionVertical"]
                ).astype(np.float32)

            # 4. Frequency Encoding for High-Cardinality IDs
            freq_cols = [
                "CityID",
                "OEMNameID",
                "OEMModelID",
                "ProcessorModelID",
                "AvSigVersion",
            ]
            for col in freq_cols:
                if col in df.columns:
                    freq = df[col].value_counts()
                    df[f"{col}_freq"] = df[col].map(freq).fillna(0).astype(np.float32)

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
            df = self.add_domain_features(df)

            # Cast text features to categorical type for LightGBM
            cat_cols = df.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                df[col] = df[col].astype("category")

            df = reduce_mem_usage(df)
            logger.info("Feature engineering completed successfully.")
            return df
        except Exception as e:
            raise ThreatForecasterException(e, sys)
