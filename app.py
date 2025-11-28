# app.py

import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Configuración básica de la página
st.set_page_config(
    page_title="Boston Housing - XGBoost", page_icon="🏠", layout="centered"
)

# ===========================
# Encabezado principal
# ===========================
st.title("🏠 Predicción de Precios de Casas - Boston")
st.markdown(
    """
Pequeña aplicación de **Machine Learning (Regresión)** construida con:

- Modelo: **XGBoost tunado**
- Target: `medv` (precio medio de viviendas en miles de dólares)
- Transformación: `log(medv)` para entrenar y luego se revierte con `exp(medv_log) - 1`

Usa la barra lateral 👉 para ingresar las características de la vivienda.
"""
)

st.markdown("---")

# ===========================
# Sidebar: parámetros de entrada
# ===========================
st.sidebar.title("⚙️ Parámetros de entrada")

st.sidebar.markdown("Introduce las variables originales del dataset de Boston Housing:")

crim = st.sidebar.number_input("crim (Tasa de criminalidad)", value=0.1, step=0.01)
zn = st.sidebar.number_input("zn (% zona residencial)", value=0.0, step=1.0)
indus = st.sidebar.number_input("indus (% acres no comerciales)", value=6.0, step=0.1)
nox = st.sidebar.number_input("nox (Óxidos de nitrógeno)", value=0.5, step=0.01)
rm = st.sidebar.number_input("rm (Nº habitaciones promedio)", value=6.0, step=0.1)
edad = st.sidebar.number_input("edad (% viviendas antiguas)", value=60.0, step=1.0)
dis = st.sidebar.number_input(
    "dis (distancia a centros de empleo)", value=4.0, step=0.1
)
rad = st.sidebar.number_input("rad (índice accesibilidad radial)", value=4, step=1)
impuesto = st.sidebar.number_input("impuesto (tasa impositiva)", value=300.0, step=1.0)
ptratio = st.sidebar.number_input(
    "ptratio (ratio alumno/profesor)", value=18.0, step=0.1
)
negro = st.sidebar.number_input("negro (índice población negra)", value=390.0, step=1.0)
lstat = st.sidebar.number_input("lstat (% bajo estatus)", value=10.0, step=0.1)

predict_button = st.sidebar.button("📈 Predecir precio")

# ===========================
# Sección central: resultado
# ===========================
st.subheader("📊 Resultado")

if predict_button:
    try:
        # Construir el objeto CustomData
        input_data = CustomData(
            crim=crim,
            zn=zn,
            indus=indus,
            nox=nox,
            rm=rm,
            edad=edad,
            dis=dis,
            rad=int(rad),
            impuesto=impuesto,
            ptratio=ptratio,
            negro=negro,
            lstat=lstat,
        )

        df_input = input_data.get_data_as_dataframe()

        # Pipeline de predicción
        predictor = PredictPipeline()
        pred = predictor.predict(df_input)  # array

        precio = float(pred[0])

        # “Tarjeta” de resultado
        st.markdown(
            f"""
        <div style="padding: 1.2rem; border-radius: 0.75rem;
                    border: 1px solid #e0e0e0;
                    background-color: #f8f9fa;">
        <h3 style="margin-bottom: 0.5rem;">💰 Precio estimado</h3>
        <p style="font-size: 1.5rem; font-weight: bold;
                    margin-bottom: 0.2rem;">
            {precio:.2f}
            <span style="font-size: 1.0rem;">miles de USD</span>
        </p>
        <p style="color: #6c757d; margin-bottom: 0;">
            El valor corresponde a la variable <code>medv</code> del
            dataset de Boston.
        </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Mostrar datos de entrada
        with st.expander("📄 Ver datos de entrada utilizados"):
            st.dataframe(df_input)

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

else:
    st.info(
        "Usa la barra lateral para ingresar los datos y presiona "
        "**📈 Predecir precio**."
    )

# ===========================
# Información del modelo
# ===========================
st.markdown("---")
st.markdown("### 🧠 Información del modelo")

st.markdown(
    """
- Algoritmo: **XGBoostRegressor**
- Target transformado: `medv_log = log1p(medv)`
- Features finales usadas:
    - `lstat_log = log1p(lstat)`
    - `crim_log = log1p(crim)`
    - `contaminacion = nox * indus`
    - `zn_log = log1p(zn)`
- Estandarización: `StandardScaler` sobre las 4 features finales
- Métricas aproximadas en test:
    - **R² ≈ 0.79**
    - **RMSE ≈ 0.17 (en escala log)**

Todo el flujo completo está implementado en:
- `src/components/data_ingestion.py`
- `src/components/data_transformation.py`
- `src/components/model_trainer.py`
- `src/pipeline/training_pipeline.py`
- `src/pipeline/predict_pipeline.py`
"""
)
