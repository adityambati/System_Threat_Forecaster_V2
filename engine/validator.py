import sys
import pandas as pd
from engine.logger import logger
from engine.exception import ThreatForecasterException


class DataValidator:
    """Validates dataset schema integrity and required features."""

    def __init__(self, config: dict):
        self.config = config

    def validate_schema(self, df: pd.DataFrame, is_training: bool = True) -> bool:
        """Checks for non-empty dataframe and target presence in training."""
        try:
            target_col = self.config["data"]["target_column"]
            id_col = self.config["data"]["id_column"]

            if is_training and target_col not in df.columns:
                logger.error(
                    f"Target column '{target_col}' missing from training dataset!"
                )
                return False

            if id_col not in df.columns:
                logger.warning(f"ID column '{id_col}' not found in dataset!")

            if df.empty:
                logger.error("Dataset provided is completely empty!")
                return False

            logger.info("Data schema validation passed successfully.")
            return True
        except Exception as e:
            raise ThreatForecasterException(e, sys)
