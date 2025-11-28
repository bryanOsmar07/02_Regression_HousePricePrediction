# src/components/model_trainer.py

import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model_xgb_tuned.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info(
                "Dividiendo train y test en X e y para el entrenamiento del modelo"
            )

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info(f"Shape X_train: {X_train.shape}, y_train: {y_train.shape}")
            logging.info(f"Shape X_test: {X_test.shape}, y_test: {y_test.shape}")

            logging.info("Iniciando entrenamiento de XGBoost Tuned")

            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_estimators=800,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.7,
                colsample_bytree=0.6,
                min_child_weight=7,
                gamma=0,
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            logging.info(
                f"XGBoost Tuned - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}"
            )

            # Guardar modelo
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=model
            )
            logging.info(
                "Modelo guardado correctamente en %s",
                self.model_trainer_config.trained_model_file_path,
            )

            return {"r2": r2, "rmse": rmse, "mae": mae}

        except Exception as e:
            raise CustomException(e, sys)
