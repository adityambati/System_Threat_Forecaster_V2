import sys
import pandas as pd
import numpy as np
from engine.logger import logger
from engine.exception import ThreatForecasterException
from engine.utils import reduce_mem_usage


class DataProcessor:
    """Advanced feature engineering: version parsing, patch deltas, hardware ratios, frequency encoding, and domain interactions."""

    def __init__(self, config: dict):
        self.config = config
        self.version_cols = config["features"]["version_columns"]
        self.date_cols = config["features"]["date_columns"]

    def parse_versions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Splits version strings (e.g. 1.1.15100.1) into numeric components and version deltas."""
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

            # Version Patch Age Deltas
            if "EngineVersion_2" in df.columns and "AppVersion_2" in df.columns:
                df["Ver_Engine_App_Delta"] = (
                    df["EngineVersion_2"] - df["AppVersion_2"]
                ).astype(np.float32)

            if (
                "OSBuildNumberOnly" in df.columns
                and "OSBuildRevisionOnly" in df.columns
            ):
                df["OS_Patch_Revision_Ratio"] = (
                    df["OSBuildNumberOnly"] / (df["OSBuildRevisionOnly"] + 1e-5)
                ).astype(np.float32)

            return df
        except Exception as e:
            raise ThreatForecasterException(e, sys)

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts numerical date components and elapsed days features."""
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

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers domain ratios, interaction features, frequency encodings, and group stats."""
        try:
            # 1. Hardware Ratios & Storage Metrics
            if (
                "TotalPhysicalRAMMB" in df.columns
                and "ProcessorCoreCount" in df.columns
            ):
                df["RAM_per_Core"] = (
                    df["TotalPhysicalRAMMB"] / (df["ProcessorCoreCount"] + 1e-5)
                ).astype(np.float32)

            if (
                "SystemVolumeCapacityMB" in df.columns
                and "PrimaryDiskCapacityMB" in df.columns
            ):
                df["Disk_Space_Ratio"] = (
                    df["SystemVolumeCapacityMB"] / (df["PrimaryDiskCapacityMB"] + 1e-5)
                ).astype(np.float32)

            if (
                "PrimaryDisplayResolutionHorizontal" in df.columns
                and "PrimaryDisplayResolutionVertical" in df.columns
            ):
                df["Screen_Pixel_Area"] = (
                    df["PrimaryDisplayResolutionHorizontal"]
                    * df["PrimaryDisplayResolutionVertical"]
                ).astype(np.float32)

            # 2. Security Configuration Vulnerability Index
            sec_cols = [
                "RealTimeProtectionState",
                "IsSMode",
                "FirewallEnabled",
                "EnableLUA",
                "IsSecureBootEnabled",
            ]
            present_sec_cols = [c for c in sec_cols if c in df.columns]
            if present_sec_cols:
                df["Security_Features_Active_Sum"] = (
                    df[present_sec_cols].fillna(0).sum(axis=1).astype(np.float32)
                )

            # 3. Categorical Interactions
            if "PlatformType" in df.columns and "Processor" in df.columns:
                df["Platform_Processor_Interact"] = (
                    df["PlatformType"].astype(str) + "_" + df["Processor"].astype(str)
                )

            if "OSEdition" in df.columns and "SKUEditionName" in df.columns:
                df["OS_SKU_Interact"] = (
                    df["OSEdition"].astype(str) + "_" + df["SKUEditionName"].astype(str)
                )

            # 4. Frequency Encoding (Both Text and High-Cardinality Numeric IDs)
            freq_cols = [
                "CityID",
                "OEMNameID",
                "OEMModelID",
                "ProcessorModelID",
                "FirmwareVersionID",
                "AntivirusConfigID",
                "CountryID",
                "GeoRegionID",
                "LocaleEnglishNameID",
                "OSBuildLab",
            ]
            for col in freq_cols:
                if col in df.columns:
                    freq = df[col].value_counts()
                    df[f"{col}_freq"] = df[col].map(freq).fillna(0).astype(np.float32)

            # 5. Group Aggregations (Deviations from OEM model averages)
            if "SystemVolumeCapacityMB" in df.columns and "OEMModelID" in df.columns:
                group_mean = df.groupby("OEMModelID")[
                    "SystemVolumeCapacityMB"
                ].transform("mean")
                df["Disk_Cap_Diff_from_OEM_Mean"] = (
                    (df["SystemVolumeCapacityMB"] - group_mean)
                    .fillna(0)
                    .astype(np.float32)
                )

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
            df = self.add_advanced_features(df)

            # Cast text/object features to categorical dtype for LightGBM
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                df[col] = df[col].astype(str).fillna("missing")
                df[col] = df[col].astype("category")

            df = reduce_mem_usage(df)
            logger.info("Feature engineering completed successfully.")
            return df
        except Exception as e:
            raise ThreatForecasterException(e, sys)
