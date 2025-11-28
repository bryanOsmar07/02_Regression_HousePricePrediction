# src/pipeline/predict_pipeline.py

import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


def _feature_engineering_new(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mismas transformaciones que en data_transformation,
    pero sin el target (no tenemos medv).
    """
    df = df.copy()
    df["lstat_log"] = np.log1p(df["lstat"])
    df["crim_log"] = np.log1p(df["crim"])
    df["contaminacion"] = df["nox"] * df["indus"]
    df["zn_log"] = np.log1p(df["zn"])

    return df[["lstat_log", "crim_log", "contaminacion", "zn_log"]]


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model_xgb_tuned.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            features_path = os.path.join("artifacts", "features.pkl")

            model = load_object(model_path)
            scaler = load_object(preprocessor_path)
            feature_list = load_object(features_path)

            # 1. Feature engineering
            feats = _feature_engineering_new(features)

            # 2. Asegurar orden y subset de columnas
            feats = feats[feature_list]

            # 3. Escalado
            feats_scaled = scaler.transform(feats)

            # 4. Predicción en escala logarítmica
            preds_log = model.predict(feats_scaled)

            # 5. Volver a escala original de precio
            preds = np.expm1(preds_log)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Clase de ayuda para construir un DataFrame a partir de inputs individuales.
    Ajusta los parámetros según las columnas originales de tu dataset.
    """

    def __init__(
        self,
        crim: float,
        zn: float,
        indus: float,
        nox: float,
        rm: float,
        edad: float,
        dis: float,
        rad: int,
        impuesto: float,
        ptratio: float,
        negro: float,
        lstat: float,
    ):
        self.crim = crim
        self.zn = zn
        self.indus = indus
        self.nox = nox
        self.rm = rm
        self.edad = edad
        self.dis = dis
        self.rad = rad
        self.impuesto = impuesto
        self.ptratio = ptratio
        self.negro = negro
        self.lstat = lstat

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data_dict = {
                "crim": [self.crim],
                "zn": [self.zn],
                "indus": [self.indus],
                "nox": [self.nox],
                "rm": [self.rm],
                "edad": [self.edad],
                "dis": [self.dis],
                "rad": [self.rad],
                "impuesto": [self.impuesto],
                "ptratio": [self.ptratio],
                "negro": [self.negro],
                "lstat": [self.lstat],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
