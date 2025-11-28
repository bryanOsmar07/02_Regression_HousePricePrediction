from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


def test_transformation_shapes():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(
        train_path, test_path
    )

    assert train_arr.shape[1] == test_arr.shape[1]  # mismas columnas
    assert train_arr.shape[1] == 5  # 4 features + target
