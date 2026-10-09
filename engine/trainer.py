import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import lightgbm as lgb
import optuna
import shap

from engine.logger import logger
from engine.exception import ThreatForecasterException
from engine.utils import save_object

# Silence Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """Handles cross-validation, LightGBM hyperparameter tuning via Optuna, model evaluation, and SHAP interpretability."""

    def __init__(self, config: dict):
        self.config = config
        self.model_dir = config["artifacts"]["model_dir"]
        self.metrics_dir = config["artifacts"]["metrics_dir"]
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def optimize_lightgbm(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int = 10
    ) -> dict:
        """Uses Optuna to tune LightGBM hyperparameters efficiently."""
        logger.info(
            f"Starting Optuna hyperparameter optimization ({n_trials} trials)..."
        )

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "max_depth": trial.suggest_int("max_depth", 5, 12),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "random_state": 42,
            }

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            auc_scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

                model = lgb.LGBMClassifier(**params, n_estimators=300)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                preds = model.predict_proba(X_va)[:, 1]
                auc_scores.append(roc_auc_score(y_val, preds))

            return np.mean(auc_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Optuna Best Trial ROC-AUC: {study.best_value:.5f}")

        # Save best parameters
        param_file = os.path.join(self.metrics_dir, "best_lightgbm_params.json")
        with open(param_file, "w") as f:
            json.dump(study.best_params, f, indent=4)

        return study.best_params

    def train_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        params: dict = None,
        n_splits: int = 5,
    ):
        """Executes Fast Stratified K-Fold CV using LightGBM."""
        try:
            logger.info(f"Initiating {n_splits}-Fold Cross-Validation with LightGBM...")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            oof_preds = np.zeros(len(X))
            test_preds = np.zeros(len(X_test))
            models = []

            default_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.03,
                "num_leaves": 127,
                "max_depth": 10,
                "min_child_samples": 30,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "random_state": 42,
                "n_estimators": 1200,
                "verbose": -1,
            }

            if params:
                default_params.update(params)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]

                model = lgb.LGBMClassifier(**default_params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )

                val_probs = model.predict_proba(X_va)[:, 1]
                oof_preds[val_idx] = val_probs
                test_preds += model.predict_proba(X_test)[:, 1] / n_splits
                models.append(model)

                fold_auc = roc_auc_score(y_val, val_probs)
                fold_acc = accuracy_score(y_val, (val_probs >= 0.5).astype(int))
                logger.info(
                    f"Fold {fold+1} | ROC-AUC: {fold_auc:.5f} | Accuracy: {fold_acc:.5f}"
                )

            overall_auc = roc_auc_score(y, oof_preds)
            overall_acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))
            logger.info(
                f"OOF Overall LightGBM ROC-AUC: {overall_auc:.5f} | Overall Accuracy: {overall_acc:.5f}"
            )

            # Save primary LightGBM model artifact
            best_model_path = os.path.join(self.model_dir, "lightgbm_model.pkl")
            save_object(best_model_path, models[0])

            # Generate evaluation plots & SHAP analysis
            self._generate_eval_artifacts(y, oof_preds, models[0], X)

            return oof_preds, test_preds, overall_auc, overall_acc

        except Exception as e:
            raise ThreatForecasterException(e, sys)

    def _generate_eval_artifacts(
        self, y_true: pd.Series, y_probs: np.ndarray, model, X_sample: pd.DataFrame
    ):
        """Generates confusion matrix and SHAP feature importance plots."""
        try:
            logger.info("Generating evaluation plots and SHAP analysis...")
            y_pred = (y_probs >= 0.5).astype(int)

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Out-of-Fold Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(os.path.join(self.metrics_dir, "confusion_matrix.png"))
            plt.close()

            # SHAP Analysis
            explainer = shap.TreeExplainer(model)
            sample_subset = X_sample.sample(min(1000, len(X_sample)), random_state=42)
            shap_values = explainer.shap_values(sample_subset)

            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], sample_subset, show=False)
            else:
                shap.summary_plot(shap_values, sample_subset, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.metrics_dir, "shap_summary.png"))
            plt.close()

        except Exception as e:
            logger.warning(f"Could not generate complete evaluation plots: {str(e)}")
