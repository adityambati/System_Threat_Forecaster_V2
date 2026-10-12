import os
import sys
import yaml
import joblib
import pandas as pd
import numpy as np
from engine.logger import logger
from engine.exception import ThreatForecasterException


def read_config(config_path: str = "config.yaml") -> dict:
    """Reads central YAML configuration file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error reading YAML config: {str(e)}")
        raise ThreatForecasterException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """Saves a Python object to disk using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object successfully saved at {file_path}")
    except Exception as e:
        logger.error(f"Failed to save object at {file_path}")
        raise ThreatForecasterException(e, sys)


def load_object(file_path: str) -> object:
    """Loads a Python object from disk."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at path: {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {file_path}")
        raise ThreatForecasterException(e, sys)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts numeric types to optimize memory usage."""
    try:
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object and col_type.name != "category":
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float32)
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
        return df
    except Exception as e:
        raise ThreatForecasterException(e, sys)
