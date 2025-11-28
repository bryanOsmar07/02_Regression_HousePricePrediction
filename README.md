# 🏠 Boston Housing Price Prediction — XGBoost (End-to-End ML Project)

Este proyecto implementa un flujo completo de **Machine Learning para regresión**, utilizando el dataset de **Boston Housing**, con un entrenamiento de modelo **XGBoost tunado**, un **pipeline modular en Python** y una **aplicación web en Streamlit** para probar predicciones en tiempo real.

El objetivo final es predecir el valor medio de viviendas (`medv`) en miles de dólares.

---

## 📌 **1. Objetivo del proyecto**

Construir un proyecto **end-to-end** que incluya:

✔ Exploración y análisis de datos (EDA)  
✔ Ingeniería de características  
✔ Selección de variables  
✔ Entrenamiento de modelos tradicionales y de boosting  
✔ Búsqueda de hiperparámetros (GridSearchCV / RandomizedSearchCV)  
✔ Validación cruzada  
✔ Pipeline modular en `.py`  
✔ Despliegue de una app en Streamlit  

---

## 📁 **2. Estructura del proyecto**

```bash
├─ data/
│  └─ bostonvivienda.csv
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  └─ 02_Modelado.ipynb
├─ src/
│  ├─ components/
│  │  ├─ data_ingestion.py
│  │  ├─ data_transformation.py
│  │  └─ model_trainer.py
│  ├─ pipeline/
│  │  ├─ training_pipeline.py
│  │  └─ predict_pipeline.py
│  ├─ utils.py
│  ├─ exception.py
│  └─ logger.py
├─ artifacts/
│  ├─ raw.csv
│  ├─ train.csv
│  ├─ test.csv
│  ├─ preprocessor.pkl
│  ├─ features.pkl
│  └─ model_xgb_tuned.pkl
├─ app.py
├─ requirements.txt
└─ README.md
```

## 📊 **3. Dataset — Boston Housing**

Cada fila representa una zona residencial en Boston.

Variables principales:

| Variable | Descripción                             |
| -------- | --------------------------------------- |
| crim     | Tasa de criminalidad                    |
| zn       | % zona residencial                      |
| indus    | % acres no comerciales                  |
| nox      | Óxidos de nitrógeno                     |
| rm       | Nº promedio de habitaciones             |
| edad     | % viviendas antiguas                    |
| dis      | Distancia ponderada a centros de empleo |
| rad      | Índice de accesibilidad radial          |
| impuesto | Tasa impositiva                         |
| ptratio  | Ratio alumno/profesor                   |
| negro    | Índice población negra                  |
| lstat    | % población con bajo estatus            |
| **medv** | **Valor medio de vivienda (target)**    |

---

## 🔍 **4. EDA – Hallazgos principales**

El análisis exploratorio incluyó:

* Distribuciones, histogramas, boxplots
* Medidas estadísticas: media, mediana, CV, asimetría, curtosis
* Detección de outliers por IQR
* Correlaciones y mapa de calor
* Relación entre cada predictor y el target

Insights clave

* Fuerte relación negativa:
    * lstat vs. medv
    * Mayor pobreza → menor precio de vivienda.

* Fuerte relación positiva:
    * rm vs. medv
    * Más habitaciones → mayor valor.

* Variables altamente correlacionadas entre sí:
    * nox, indus, rad, impuesto

* Variables con fuerte asimetría:
    * crim, zn, lstat

🔹 El dataset contiene varios outliers naturales — no se eliminaron para mantener consistencia histórica.

---

## 🧪 **5. Ingeniería de Variables**

Para mejorar la simetría, estabilizar la varianza y capturar relaciones no lineales, se aplicaron:

```python
lstat_log      = log1p(lstat)
crim_log       = log1p(crim)
contaminacion  = nox * indus
zn_log         = log1p(zn)
medv_log       = log1p(medv)
```

✔ Se eligió trabajar con medv_log como target transformado.
✔ Luego se revierte usando exp(pred) - 1.

---

## 🧬 **6. Selección de Variables**

* Se utilizó una combinación de:
* Correlación con el target
* VIF (multicolinealidad)
* Interpretabilidad
* Pruebas de modelado

**Variables finales seleccionadas:**

* lstat_log  
* crim_log  
* contaminacion  
* zn_log

Estas 4 variables lograron capturar >80% del poder predictivo del modelo original.

---

## 🤖 **7. Modelado y Evaluación**

Se probaron múltiples modelos:

* Linear Regression
* Ridge / Lasso / ElasticNet
* RandomForest
* XGBoost (mejor modelo)

🔥 **Métricas obtenidas (test)**

| Modelo            | MAE       | RMSE      | R²       |
| ----------------- | --------- | --------- | -------- |
| Linear Regression | 0.147     | 0.207     | 0.69     |
| Ridge             | 0.147     | 0.207     | 0.69     |
| RandomForest      | 0.127     | 0.169     | 0.79     |
| **XGBoost Tuned** | **0.125** | **0.170** | **0.79** |


---

## 🛠 **8. Tuning de Hiperparámetros (GridSearchCV)**

Mejores hiperparámetros encontrados:

```python
{
 'n_estimators': 800,
 'max_depth': 5,
 'learning_rate': 0.01,
 'subsample': 0.7,
 'colsample_bytree': 0.6,
 'min_child_weight': 7,
 'gamma': 0
}
```

✔ Aumento de performance

✔ Reducción de overfitting

✔ Mejor estabilidad entre folds de validación cruzada

---

## 📦 **9. Pipeline en scripts (.py)**

**data_ingestion.py**
* Lee dataset
* Crea raw/train/test
* Loguea eventos

**data_transformation.py**
* Aplica feature engineering
* Estandariza variables
* Guarda scaler y lista de features

**model_trainer.py**
* Entrena XGBoost tunado
* Evalúa métricas R², RMSE, MAE
* Guarda modelo final

**training_pipeline.py**
* Orquesta todo el proceso:
    * Ingesta → Transformación → Entrenamiento → Guardado de artifacts

**predict_pipeline.py**
* Feature engineering para nuevos datos
* Escalamiento
* Predicción
* Reversión logarítmica

---

## 🌐 **10. Aplicación Web — Streamlit**

La app permite ingresar 12 variables y devuelve el precio estimado.

**Ejecutar aplicación:**

```bash
streamlit run app.py
```

¿Qué incluye?

✔ Formulario intuitivo

✔ “Tarjeta” con resultado

✔ Expander con datos usados

✔ Explicación del modelo

✔ UI moderna

---

## ⚙️ **11. Cómo ejecutar el proyecto**
1️⃣ **Crear entorno**

```bash
conda create -p venv python=3.8 -y
conda activate venv
```

2️⃣ **Instalar dependencias**

```bash
pip install -r requirements.txt
```

3️⃣ **Entrenar el pipeline**

```bash
python src/pipeline/training_pipeline.py
```

4️⃣ **Ejecutar app Streamlit**

```bash
streamlit run app.py
```

---

## 🧪 **12. Usar el modelo desde Python**

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData(
    crim=0.1,
    zn=10,
    indus=5.0,
    nox=0.5,
    rm=6.5,
    edad=60,
    dis=4.0,
    rad=4,
    impuesto=300,
    ptratio=18,
    negro=390,
    lstat=12
)

df = data.get_data_as_dataframe()
pred = PredictPipeline().predict(df)

print(pred[0])
```
---

## 🚀 **13. Mejoras futuras**

* Despliegue en Streamlit Cloud / Render / Hugging Face
* Explicabilidad del modelo (SHAP)
* API REST con FastAPI
* MLflow para tracking de experimentos
* Validación de inputs con Pydantic


## ✅ **14. Integración continua (CI) con GitHub Actions**

Este repositorio incluye un workflow de GitHub Actions (`.github/workflows/ci.yml`) que ejecuta automáticamente:

- Instalación de dependencias
- Revisión de estilo de código:
  - `isort` (orden de imports)
  - `black` (formato de código)
  - `flake8` (linting)
- Tests unitarios con `pytest`

El workflow se ejecuta en cada **push** y **pull request** a la rama `main`, verificando que:

- El código siga estándares de calidad
- Los tests pasen correctamente
- El proyecto sea estable antes de mezclar cambios

Esto imita un entorno real de trabajo con **CI**.

![CI](https://github.com/bryanOsmar07/02_Regression_HousePricePrediction/actions/workflows/ci.yml/badge.svg)

📦 **Instalación de dependencias de desarrollo**

```bash
pip install -r requirements-dev.txt

black
flake8
isort
pytest
pre-commit
```

🪝 **Pre-commit Hooks**

Para garantizar un código limpio en cada commit:

```bash
pre-commit install
```
Ejecutar manualmente sobre todos los archivos:

```bash
pre-commit run --all-files
```

Estos hooks aseguran que no puedas hacer commit si el código no cumple estándares.


## 🧪 **15. Pruebas unitarias (Pytest)**

El proyecto incluye tests para:

* Ingesta de datos
* Transformación
* Entrenamiento de modelo
* Pipeline de predicción
* Funciones utilitarias

Para ejecutar:

```bash
pytest -v
```

## 🚀 **16. Integración continua (CI/CD) con GitHub Actions**

Este repositorio usa un workflow en:

```bash
.github/workflows/ci.yml
```

El workflow se ejecuta automáticamente en cada push y pull request a main.

¿Qué valida?

✔ Instalación del proyecto

✔ Linting (black, isort, flake8)

✔ Pruebas unitarias con pytest

✔ Garantiza que el proyecto no se rompa

Badge (opcional)

```md
![CI](https://github.com/bryanOsmar07/02_Regression_HousePricePrediction/actions/workflows/ci.yml/badge.svg)
```

## 🖥️ **17. Ejecutar la app sin usar terminal**

✔ Opción 1 — Archivo .bat (Windows)

Crear run_app.bat:

```bat
@echo off
cd /d %~dp0

call venv\Scripts\activate

streamlit run app.py

pause
```

## 🏁 **18. Estado del proyecto**

✔ End-to-end pipeline

✔ Modelo XGBoost tunado

✔ Linting / Testing / Pre-commit

✔ CI/CD con GitHub Actions

✔ App Streamlit totalmente funcional

✔ Ejecutable con un clic

👨‍💻 Autor

Proyecto desarrollado por Brayan Osmar Quispe Montoya
Data Scientist
2025