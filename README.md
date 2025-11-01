# 📊 Análisis del Mercado Laboral de Data Science en España (2025)

> Proyecto completo de análisis, modelado y visualización del mercado laboral en el sector **Data Science / IA / Machine Learning** en España.  
Incluye scraping de ofertas reales, análisis exploratorio, modelado predictivo y dashboard interactivo con Streamlit.

---

## 🚀 Demo del Proyecto

🔗 **[Ver Dashboard en Streamlit Cloud](https://pabloramos-ds-job-market.streamlit.app/)** *(ejemplo de enlace)*  
📈 **[Ver experimentos en Weights & Biases](https://wandb.ai/paulsbusiness1111/data-science-job-market-es)**

---

## 🧠 Objetivos del Proyecto

- Analizar el estado actual del mercado laboral en **Data Science en España**.  
- Identificar los **roles, skills y ubicaciones más demandadas**.  
- Estimar rangos salariales aproximados mediante **modelos predictivos**.  
- Crear un **dashboard interactivo** de exploración de datos.

---

## 🏗️ Estructura del Proyecto

data-science-job-market-spain/
│
├── app/
│ └── streamlit_dashboard.py # Dashboard interactivo
│
├── data/
│ ├── raw/ # Datos sin procesar
│ └── processed/ # Datos limpios (jobs_cleaned_cleaned.csv)
│
├── models/
│ ├── salary_predictor.pkl
│ ├── scaler.pkl
│ └── model_metadata.json
│
├── src/
│ ├── data_collection.py # Extracción desde Adzuna API
│ ├── data_cleaning.py # Limpieza y normalización
│ ├── feature_engineering.py # Ingeniería de variables
│ ├── model.py # Entrenamiento y evaluación del modelo
│ └── fix_locations.py # Limpieza avanzada de ubicaciones
│
├── images/ # Gráficos generados y capturas
│
├── requirements.txt
└── README.md


---

## ⚙️ Tecnologías y Librerías

| Tipo | Herramientas |
|------|---------------|
| 📦 Extracción de datos | Adzuna API, requests |
| 🧹 Limpieza / EDA | pandas, numpy, matplotlib, seaborn |
| 🤖 Modelado | scikit-learn (Ridge, RandomForest, GradientBoosting) |
| 🧠 Seguimiento de experimentos | Weights & Biases (wandb) |
| 📊 Dashboard | Streamlit, Plotly |
| 💾 Serialización | joblib, json |

---

## 🧩 Pipeline del Proyecto

1. **Recolección de Datos:**  
   Scraping de más de **2 000 ofertas reales** del portal Adzuna (España).

2. **Limpieza e Ingeniería de Features:**  
   - Normalización de ciudades y roles.  
   - Extracción de skills más frecuentes.  
   - Cálculo del salario medio (`salary_avg`).

3. **Análisis Exploratorio (EDA):**  
   Visualizaciones sobre demanda, localización y habilidades clave.

4. **Modelado Predictivo:**  
   Comparación de Ridge, RandomForest y Gradient Boosting →  
   ✅ **Mejor modelo:** Ridge Regression (R² ≈ 0.008, MAE ≈ 9 734 €).

5. **Dashboard Interactivo:**  
   Navegación por pestañas:
   - 🏠 Overview general  
   - 🗺️ Análisis geográfico  
   - 🔥 Skills demandadas  
   - 💰 Salarios  
   - 🤖 IA/ML Trends  
   - 🔮 Predictor de salarios

---

## 📈 Resultados Clave

| Insight | Resultado |
|----------|------------|
| 📍 Ciudades con más ofertas | Madrid, Barcelona, Valencia |
| 💼 Roles más demandados | Data Analyst, Data Engineer, Data Scientist |
| 🔥 Skills top | Python, SQL, Machine Learning, AWS, Power BI |
| 💰 Salario medio estimado | ~42 000 €/año |
| 🤖 % de ofertas IA/ML | 25 % del total aproximado |

---

## ⚙️ Instalación Local

```bash
# 1️⃣ Clonar el repositorio
git clone https://github.com/<tu_usuario>/data-science-job-market-spain.git
cd data-science-job-market-spain

# 2️⃣ Crear entorno virtual
python -m venv venv
source venv/bin/activate   # o venv\Scripts\activate en Windows

# 3️⃣ Instalar dependencias
pip install -r requirements.txt

# 4️⃣ Ejecutar dashboard
streamlit run app/streamlit_dashboard.py


Proyecto desarrollado como parte de portfolio en Data Science e Inteligencia Artificial (2025).
Dataset obtenido mediante la API pública de Adzuna.