import sys
import pandas as pd
from engine.utils import read_config, load_object
from engine.data_processor import DataProcessor
from engine.logger import logger
from engine.exception import ThreatForecasterException


class PredictPipeline:
    """Handles production inference on unseen data using saved model binaries."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = read_config(config_path)
        self.processor = DataProcessor(self.config)
        self.model_path = f"{self.config['artifacts']['model_dir']}lightgbm_model.pkl"

    def predict(self, input_df: pd.DataFrame) -> pd.Series:
        try:
            logger.info("Starting inference pipeline execution...")
            df = input_df.copy()

            id_col = self.config["data"]["id_column"]
            target_col = self.config["data"]["target_column"]

            if id_col in df.columns:
                df.drop(columns=[id_col], inplace=True)
            if target_col in df.columns:
                df.drop(columns=[target_col], inplace=True)

            # Feature Engineering
            X_transformed = self.processor.transform(df)

            # Load Trained Model
            model = load_object(self.model_path)

            # Predict Probabilities
            probabilities = model.predict_proba(X_transformed)[:, 1]
            logger.info("Inference successfully executed.")
            return probabilities

        except Exception as e:
            raise ThreatForecasterException(e, sys)
