import sys
import pandas as pd
from engine.utils import read_config
from engine.validator import DataValidator
from engine.data_processor import DataProcessor
from engine.trainer import ModelTrainer
from engine.logger import logger
from engine.exception import ThreatForecasterException


class TrainPipeline:
    """Orchestrates full model training workflow."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = read_config(config_path)
        self.validator = DataValidator(self.config)
        self.processor = DataProcessor(self.config)
        self.trainer = ModelTrainer(self.config)

    def run(self):
        try:
            logger.info("=== Starting Training Pipeline Execution ===")

            # 1. Load Raw Datasets
            train_path = self.config["data"]["train_path"]
            test_path = self.config["data"]["test_path"]
            logger.info(
                f"Loading train data from {train_path} and test data from {test_path}..."
            )

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 2. Schema Quality Gate
            if not self.validator.validate_schema(train_df, is_training=True):
                raise ValueError("Training dataset failed schema validation checks.")

            # 3. Separate Targets & Identifiers
            target_col = self.config["data"]["target_column"]
            id_col = self.config["data"]["id_column"]

            y = train_df[target_col]
            X_train = train_df.drop(columns=[target_col])
            X_test = test_df.copy()

            test_ids = X_test[id_col] if id_col in X_test.columns else None

            if id_col in X_train.columns:
                X_train.drop(columns=[id_col], inplace=True)
            if id_col in X_test.columns:
                X_test.drop(columns=[id_col], inplace=True)

            # 4. Feature Engineering & Preprocessing
            X_train_transformed = self.processor.transform(X_train)
            X_test_transformed = self.processor.transform(X_test)

            # Align feature columns between train and test
            X_train_transformed, X_test_transformed = X_train_transformed.align(
                X_test_transformed, join="left", axis=1
            )

            # 5. Hyperparameter Tuning & Cross-Validation Training
            logger.info("Running Optuna hyperparameter optimization...")
            best_params = self.trainer.optimize_lightgbm(
                X_train_transformed, y, n_trials=10
            )

            logger.info("Executing tuned model training and cross-validation...")
            oof_preds, test_preds, oof_auc, oof_acc = (
                self.trainer.train_cross_validation(
                    X_train_transformed, y, X_test_transformed, params=best_params
                )
            )

            # 6. Save Submission Artifact
            submission = pd.DataFrame()
            if test_ids is not None:
                submission[id_col] = test_ids
            submission[target_col] = test_preds

            submission_path = "submission.csv"
            submission.to_csv(submission_path, index=False)
            logger.info(f"Submission saved successfully to {submission_path}")

            print(f"\n==========================================")
            print(f" TRAINING COMPLETE")
            print(f" Overall OOF ROC-AUC : {oof_auc:.5f}")
            print(f" Overall Accuracy    : {oof_acc:.5f}")
            print(f" Output Saved To     : {submission_path}")
            print(f"==========================================\n")

        except Exception as e:
            raise ThreatForecasterException(e, sys)
