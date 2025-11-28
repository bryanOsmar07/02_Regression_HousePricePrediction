# src/pipeline/training_pipeline.py

import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def run_training_pipeline():
    try:
        logging.info("======== INICIO TRAINING PIPELINE ========")

        # 1. Ingesta de datos
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info(f"Rutas generadas - train: {train_path}, test: {test_path}")

        # 2. Transformación
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(
            train_path, test_path
        )
        logging.info(
            f"Shapes finales - train_arr: {train_arr.shape}, test_arr: {test_arr.shape}"
        )

        # 3. Entrenamiento de modelo
        trainer = ModelTrainer()
        metrics = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(
            f"Training Pipeline finalizado. "
            f"R2 en test: {metrics['r2']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f}, "
            f"MAE: {metrics['mae']:.4f}"
        )
        logging.info("======== FIN TRAINING PIPELINE ========")

        return metrics

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
