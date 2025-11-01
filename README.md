# 📊 Análisis del Mercado Laboral de Data Science en España (2025)
## 👨‍💻 Autor

**Pablo Iriibarren**  
💼 [LinkedIn](https://www.linkedin.com/in/pablo-iribarren-muru-93b547269) 
✉️ [Email me](mailto:pabloiribarren2003@gmail.com) 

> Proyecto completo de análisis, modelado y visualización del mercado laboral en el sector **Data Science / IA / Machine Learning** en España.  
Incluye scraping de ofertas reales, análisis exploratorio, modelado predictivo y dashboard interactivo con Streamlit.

---

## 🚀 Demo del Proyecto

🔗 **[Ver Dashboard en Streamlit Cloud](https://data-science-job-market-spain-pablooiiribarren.streamlit.app/)**
📈 **[Ver experimentos en Weights & Biases](https://wandb.ai/paulsbusiness1111/data-science-job-market-es)**

---

## 🧠 Objetivos del Proyecto

- Analizar el estado actual del mercado laboral en **Data Science en España**.  
- Identificar los **roles, skills y ubicaciones más demandadas**.  
- Estimar rangos salariales aproximados mediante **modelos predictivos**.  
- Crear un **dashboard interactivo** de exploración de datos.

---

## 🚀 Estructura del Proyecto
data-science-job-market-spain/
├── data/
│   ├── raw/              # Datos sin procesar (originales de la API)
│   ├── processed/        # Datos limpios y listos para análisis
│   └── sample/           # Datos de ejemplo o prueba
├── src/
│   ├── data_collection.py    # Recolección de datos desde Adzuna API
│   ├── data_processing.py    # Limpieza y preprocesamiento
│   ├── skills_extractor.py   # Extracción y análisis de habilidades
│   ├── model.py              # Entrenamiento del modelo predictivo
│   └── visualization.py      # Visualizaciones con Plotly
├── app/
│   └── streamlit_dashboard.py   # Dashboard interactivo
├── models/                 # Modelos entrenados (.pkl, .json)
├── notebooks/              # Análisis exploratorios (Jupyter)
├── images/                 # Gráficos y capturas para el README
├── requirements.txt
├── .gitignore
└── README.md
---

## ⚙️ Tecnologías y Librerías

| Categoría | Librerías |
|------------|------------|
| 📦 Extracción de datos | `requests`, `beautifulsoup4`, `selenium` |
| 🧹 Limpieza / EDA | `pandas`, `numpy` |
| 🤖 Modelado | `scikit-learn`, `xgboost` |
| 🧠 Seguimiento de experimentos | `wandb` |
| 📊 Visualización | `plotly`, `matplotlib`, `seaborn`, `wordcloud` |
| 🖥 Dashboard | `streamlit` |
| 🧰 Utilidades | `tqdm`, `dotenv`, `nltk` |

---

## 🧩 Pipeline del Proyecto

1. **Recolección de Datos:**  
   Scraping de más de **2 000 ofertas reales** de empleo de la API pública de Adzuna (España).

2. **Limpieza e Ingeniería de Features:**  
   - Normalización de nombres de ciudades y roles.  
   - Extracción de las 20 skills más frecuentes.  
   - Cálculo del salario medio (`salary_avg`).

3. **Análisis Exploratorio (EDA):**  
   Visualizaciones sobre demanda, localización y habilidades clave en el mercado.

4. **Modelado Predictivo:**  
   Comparación de modelos:
   - Ridge Regression  
   - Random Forest  
   - Gradient Boosting  
   ✅ Mejor modelo: **Ridge Regression**  
   *(R² = 0.008, MAE ≈ 9 734 €)*

5. **Dashboard Interactivo:**  
   Navegación por pestañas:
   - 🏠 **Overview general**  
   - 🗺️ **Análisis geográfico**  
   - 🔥 **Skills demandadas**  
   - 💰 **Salarios**  
   - 🤖 **Tendencias IA/ML**  
   - 🔮 **Predictor de salarios**

---

## 📈 Resultados Clave

| Insight | Resultado |
|----------|------------|
| 📍 Ciudades con más ofertas | Madrid, Barcelona, Valencia |
| 💼 Roles más demandados | Data Analyst, Data Engineer, Data Scientist |
| 🔥 Skills top | Python, SQL, Machine Learning, AWS, Power BI |
| 💰 Salario medio estimado | ~42 000 €/año |
| 🤖 % de ofertas IA/ML | 25 % aprox. del total |

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
