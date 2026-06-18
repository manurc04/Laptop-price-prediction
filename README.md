# 💻 Predicción de precios de portátiles

Proyecto de **Data Science y Machine Learning** para la predicción de precios: empezando por la
descarga y limpieza de datos, pasando por el análisis exploratorio, hasta el
entrenamiento, comparación y despliegue de un modelo de regresión que estima
el precio de un portátil a partir de sus características técnicas (CPU, RAM,
GPU, almacenamiento, pantalla, etc.).


## Problema de negocio

Una tienda online o un comparador de precios necesita estimar de forma
automática si el precio de un portátil está dentro de rango de mercado según
sus especificaciones para, por ejemplo, detectar anomalías de pricing, sugerir precios a
vendedores y analizar qué características aportan más valor percibido.

## Stack técnico

- **Python 3.10+**
- `pandas`, `numpy` — manipulación de datos
- `matplotlib`, `seaborn` — visualización
- `scikit-learn`, `xgboost` — modelado y pipelines de preprocesado
- `joblib` — serialización del modelo
- `streamlit` — demo interactiva
- `pytest` — tests unitarios
- GitHub Actions — integración continua (CI)

## Estructura del proyecto

```
.
├── data/
│   ├── generate_dataset.py        # genera un dataset SINTÉTICO equivalente
│   ├── prepare_kaggle_dataset.py  # limpia y adapta el dataset REAL de Kaggle
│   └── raw/laptops.csv            # dataset final usado (real, de Kaggle)
├── notebooks/
│   └── 01_eda.ipynb          # análisis exploratorio completo
├── src/
│   ├── data_processing.py    # carga, limpieza y preprocesado
│   ├── train_model.py        # entrena y compara varios modelos
│   └── predict.py            # carga el modelo y predice
├── app/
│   └── streamlit_app.py      # demo interactiva
├── tests/
│   ├── test_data_processing.py
│   └── test_predict.py
├── models/                   # modelo entrenado (se genera, no se versiona)
├── .github/workflows/ci.yml  # pipeline de CI (tests + entrenamiento)
├── requirements.txt
└── README.md
```

## Sobre el dataset

El dataset usado es real: **"Laptop Price Prediction"** (Eslam Elsolia,
Kaggle), 1303 portátiles con especificaciones técnicas y precio.

Los datos crudos de Kaggle vienen "sucios" (texto mezclado con números,
varias informaciones en una misma columna). El script
`data/prepare_kaggle_dataset.py` se encarga de:

- Convertir `"8GB"` → `8`, `"1.37kg"` → `1.37` (extraer números de texto).
- Separar `"128GB SSD + 1TB HDD"` en capacidad (GB) y tipo de disco.
- Limpiar la CPU (quitar la velocidad en GHz) y quedarse con la marca de GPU.
- Extraer la resolución numérica y detectar si tiene pantalla táctil.
- Convertir el precio de rupias indias a dólares (tipo de cambio aproximado).

El proyecto también incluye `data/generate_dataset.py`, que genera un
dataset equivalente (útil para reproducir el pipeline sin
depender de descargar nada de Kaggle).

Para reconstruir el dataset real desde cero:

```bash
# 1. Descarga "laptop_data.csv" del dataset de Kaggle y colócalo en data/raw/
# 2. Ejecuta:
python data/prepare_kaggle_dataset.py
```

## 🚀 Cómo ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/<nombre-repo>.git
cd <nombre-repo>

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. (Opcional) Regenerar el dataset
python data/generate_dataset.py

# 4. Entrenar y comparar modelos
python -m src.train_model

# 5. Ejecutar los tests
pytest -q

# 6. Lanzar la demo interactiva
streamlit run app/streamlit_app.py
```

## Resultados del modelado

Se entrenaron y compararon 5 modelos sobre un split 80/20:

| Modelo            | MAE ($) | RMSE ($) | R²     |
|-------------------|---------|----------|--------|
| **XGBoost**       | **163.34** | **269.80** | **0.7919** |
| RandomForest      | 153.66  | 270.65   | 0.7905 |
| GradientBoosting  | 174.52  | 278.38   | 0.7784 |
| Ridge             | 182.33  | 280.45   | 0.7751 |
| LinearRegression  | 185.83  | 292.46   | 0.7554 |

Con datos reales, los modelos basados en árboles (XGBoost, RandomForest)
superan a los modelos lineales. Lo que es una señal de que las relaciones entre
especificaciones y precio no son puramente aditivas (por ejemplo, el efecto
de la RAM en el precio no es el mismo en un Chromebook que en una
Workstation).

##  Principales hallazgos del EDA (`notebooks/01_eda.ipynb`)

- La marca, la CPU y la GPU son las variables categóricas con mayor impacto
  en el precio mediano.
- RAM y almacenamiento correlacionan positivamente con el precio.
- La distribución del precio tiene cola hacia la derecha por los modelos de
  gama muy alta (gaming por ejemplo).

## Próximas mejoras

- Añadir *hyperparameter tuning* con `GridSearchCV` / `Optuna`.
- Desplegar la app de Streamlit en Streamlit Community Cloud.
- Añadir explicabilidad del modelo con SHAP.
- Enriquecer el dataset con variables actuales (año de lanzamiento, batería)
  vía scraping, ya que el dataset de Kaggle es de 2018-2019.

## Licencia

Este proyecto está bajo licencia MIT — ver [LICENSE](LICENSE).

## Autor

**Manuel Rodríguez Cristóbal** — [manuelrc100@gmail.com](mailto:manuelrc100@gmail.com)
