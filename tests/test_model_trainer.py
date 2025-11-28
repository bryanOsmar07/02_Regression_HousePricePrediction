from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def test_model_training_returns_r2():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()
    result = trainer.initiate_model_trainer(train_arr, test_arr)

    assert "r2" in result
    assert isinstance(result["r2"], float)
