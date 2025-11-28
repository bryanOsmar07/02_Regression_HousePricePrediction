# tests/test_predict_pipeline.py

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


def test_prediction_pipeline_runs():
    # Datos mínimos válidos
    data = CustomData(
        crim=0.1,
        zn=10,
        indus=5,
        nox=0.5,
        rm=6.0,
        edad=60,
        dis=4.0,
        rad=4,
        impuesto=300,
        ptratio=18,
        negro=390,
        lstat=10,
    )

    df = data.get_data_as_dataframe()

    pipeline = PredictPipeline()
    pred = pipeline.predict(df)

    # Verificar que devuelve una predicción válida
    assert pred is not None
    assert len(pred) == 1
    assert pred[0] > 0
