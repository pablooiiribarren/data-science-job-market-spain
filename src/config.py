"""
Configuración central del proyecto
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas del proyecto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Crear directorios si no existen
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Credentials
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY")

# Configuración de búsqueda
CITIES = {
    "pamplona": {"country": "es", "location": "Pamplona"},
    "san_sebastian": {"country": "es", "location": "Donostia-San Sebastián"},
    "bilbao": {"country": "es", "location": "Bilbao"}
}

SEARCH_TERMS = [
    "Data Scientist",
    "Data Analyst",
    "Machine Learning Engineer",
    "AI Engineer",
    "Data Engineer",
    "Business Intelligence",
    "ML Engineer"
]

# Skills a buscar en las descripciones
SKILLS_LIST = [
    # Lenguajes de programación
    "Python", "R", "SQL", "Java", "Scala", "Julia",
    
    # Frameworks ML/DL
    "TensorFlow", "PyTorch", "Keras", "scikit-learn", "XGBoost",
    
    # Big Data
    "Spark", "Hadoop", "Kafka", "Airflow",
    
    # Bases de datos
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Cassandra",
    
    # Cloud
    "AWS", "Azure", "GCP", "Google Cloud",
    
    # BI Tools
    "Tableau", "Power BI", "Looker", "Qlik",
    
    # Otros
    "Docker", "Kubernetes", "Git", "Linux",
    "Pandas", "NumPy", "Matplotlib", "Seaborn",
    "NLP", "Computer Vision", "Deep Learning",
    "MLOps", "CI/CD"
]

# Configuración del modelo
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# Configuración de visualización
PLOT_CONFIG = {
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "template": "plotly_white",
    "height": 500,
    "width": 800
}