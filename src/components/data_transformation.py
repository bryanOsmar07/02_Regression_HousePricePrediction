# src/components/data_transformation.py

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    features_obj_file_path: str = os.path.join("artifacts", "features.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # Variables finales definidas en el EDA
        self.feature_cols = ["lstat_log", "crim_log", "contaminacion", "zn_log"]
        self.target_col = "medv_log"

    @staticmethod
    def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replica las transformaciones del notebook EDA:
        - lstat_log, crim_log, contaminacion, zn_log, medv_log
        """
        df = df.copy()

        df["lstat_log"] = np.log1p(df["lstat"])
        df["crim_log"] = np.log1p(df["crim"])
        df["contaminacion"] = df["nox"] * df["indus"]
        df["zn_log"] = np.log1p(df["zn"])
        df["medv_log"] = np.log1p(df["medv"])

        cols_finales = ["lstat_log", "crim_log", "contaminacion", "zn_log", "medv_log"]
        return df[cols_finales]

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("Iniciando Data Transformation")
        try:
            # 1. Lectura de train y test
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Train shape original: {train_df.shape}")
            logging.info(f"Test shape original: {test_df.shape}")

            # 2. Feature engineering
            logging.info("Aplicando feature engineering a train y test")
            train_feat = self._feature_engineering(train_df)
            test_feat = self._feature_engineering(test_df)

            # 3. Separar X / y
            X_train = train_feat[self.feature_cols]
            y_train = train_feat[self.target_col]

            X_test = test_feat[self.feature_cols]
            y_test = test_feat[self.target_col]

            logging.info(f"Shape X_train antes de imputar/escalar: {X_train.shape}")
            logging.info(f"Shape X_test antes de imputar/escalar: {X_test.shape}")

            # Manejo básico de nulos
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())

            # 4. Estandarización
            scaler = StandardScaler()
            logging.info("Ajustando StandardScaler con datos de entrenamiento")
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 5. Guardar scaler
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.preprocessor_obj_file_path
                ),
                exist_ok=True,
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=scaler,
            )
            logging.info(
                f"Preprocessor (scaler) guardado en {self.data_transformation_config.preprocessor_obj_file_path}"
            )

            # 6. Guardar lista de features
            save_object(
                file_path=self.data_transformation_config.features_obj_file_path,
                obj=self.feature_cols,
            )
            logging.info(
                f"Lista de features guardada en {self.data_transformation_config.features_obj_file_path}"
            )

            # 7. Unir X e y en un solo array para pasarlo al trainer
            train_arr = np.c_[X_train_scaled, y_train.to_numpy()]
            test_arr = np.c_[X_test_scaled, y_test.to_numpy()]

            logging.info(f"Shape final train_arr: {train_arr.shape}")
            logging.info(f"Shape final test_arr: {test_arr.shape}")
            logging.info("Data Transformation finalizado correctamente")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
