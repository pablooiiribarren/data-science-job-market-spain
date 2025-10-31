"""
Script de limpieza y procesamiento de datos
Convierte datos crudos en un dataset limpio y estructurado
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
from datetime import datetime
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SKILLS_LIST

class DataProcessor:
    """
    Clase para limpiar y procesar datos de ofertas de trabajo
    """
    
    def __init__(self, input_file=None):
        """
        Inicializa el procesador
        
        Args:
            input_file: Ruta al archivo CSV crudo (si None, usa el más reciente)
        """
        if input_file is None:
            # Buscar el archivo más reciente
            files = sorted(RAW_DATA_DIR.glob("jobs_data_*.csv"))
            if not files:
                raise FileNotFoundError("No se encontraron archivos en data/raw/")
            input_file = files[-1]
        
        self.input_file = Path(input_file)
        self.df = None
        
        # Skills a buscar
        self.SKILLS = {
            'Python': r'\bpython\b',
            'R': r'\b r\b|\br programming\b',
            'SQL': r'\bsql\b',
            'Java': r'\bjava\b',
            'Scala': r'\bscala\b',
            'TensorFlow': r'\btensorflow\b',
            'PyTorch': r'\bpytorch\b',
            'Keras': r'\bkeras\b',
            'scikit-learn': r'\bscikit.learn\b|\bsklearn\b',
            'XGBoost': r'\bxgboost\b',
            'Spark': r'\bspark\b',
            'Hadoop': r'\bhadoop\b',
            'Kafka': r'\bkafka\b',
            'Airflow': r'\bairflow\b',
            'PostgreSQL': r'\bpostgresql\b|\bpostgres\b',
            'MySQL': r'\bmysql\b',
            'MongoDB': r'\bmongodb\b',
            'Elasticsearch': r'\belasticsearch\b',
            'AWS': r'\baws\b',
            'Azure': r'\bazure\b',
            'GCP': r'\bgcp\b|\bgoogle cloud\b',
            'Tableau': r'\btableau\b',
            'Power BI': r'\bpower bi\b|\bpowerbi\b',
            'Looker': r'\blooker\b',
            'Docker': r'\bdocker\b',
            'Kubernetes': r'\bkubernetes\b|\bk8s\b',
            'Git': r'\bgit\b',
            'Pandas': r'\bpandas\b',
            'NumPy': r'\bnumpy\b',
        }
        
        # Keywords de IA/ML
        self.AI_KEYWORDS = [
            'artificial intelligence', 'ai', 'machine learning', 'ml',
            'deep learning', 'neural network', 'llm', 'gpt', 'chatgpt',
            'generative ai', 'computer vision', 'nlp', 'natural language'
        ]
    
    def load_data(self):
        """Carga los datos crudos"""
        print(f"📂 Cargando: {self.input_file.name}")
        self.df = pd.read_csv(self.input_file)
        print(f"✅ {len(self.df)} ofertas cargadas")
        return self
    
    def remove_duplicates(self):
        """Elimina ofertas duplicadas"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['id'], keep='first')
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"🗑️  {removed} duplicados eliminados")
        return self
    
    def clean_locations(self):
        """Limpia y estandariza ubicaciones"""
        def clean_location(location):
            if pd.isna(location):
                return 'Desconocido'
            
            location = str(location).strip()
            
            if ',' in location:
                location = location.split(',')[0].strip()
            
            replacements = {
                'Donostia': 'San Sebastián',
                'Donostia-San Sebastián': 'San Sebastián',
                'A Coruña': 'La Coruña',
                'Vizcaya': 'Bilbao',
                'Guipúzcoa': 'San Sebastián'
            }
            
            for old, new in replacements.items():
                if old in location:
                    location = new
            
            return location
        
        self.df['city'] = self.df['location_display'].apply(clean_location)
        print(f"📍 {self.df['city'].nunique()} ciudades únicas")
        return self
    
    def extract_seniority(self):
        """Extrae nivel de experiencia del título"""
        def get_seniority(title):
            if pd.isna(title):
                return 'No especificado'
            
            title_lower = str(title).lower()
            
            if any(word in title_lower for word in ['junior', 'jr', 'trainee', 'graduate', 'entry']):
                return 'Junior'
            elif any(word in title_lower for word in ['senior', 'sr', 'lead', 'principal', 'staff']):
                return 'Senior'
            elif any(word in title_lower for word in ['manager', 'head', 'director', 'chief']):
                return 'Manager'
            else:
                return 'Mid-Level'
        
        self.df['seniority'] = self.df['title'].apply(get_seniority)
        print(f"📊 Niveles: {self.df['seniority'].value_counts().to_dict()}")
        return self
    
    def categorize_roles(self):
        """Categoriza tipos de roles"""
        def get_category(title):
            if pd.isna(title):
                return 'Otros'
            
            title_lower = str(title).lower()
            
            if 'scientist' in title_lower:
                return 'Data Scientist'
            elif 'analyst' in title_lower or 'analytics' in title_lower:
                return 'Data Analyst'
            elif 'engineer' in title_lower and ('data' in title_lower or 'ml' in title_lower):
                return 'Data/ML Engineer'
            elif 'machine learning' in title_lower or 'ml ' in title_lower:
                return 'Machine Learning'
            elif 'business intelligence' in title_lower or 'bi ' in title_lower:
                return 'Business Intelligence'
            elif 'ai' in title_lower or 'artificial intelligence' in title_lower:
                return 'AI Specialist'
            else:
                return 'Otros'
        
        self.df['role_category'] = self.df['title'].apply(get_category)
        print(f"💼 Categorías: {self.df['role_category'].value_counts().to_dict()}")
        return self
    
    def process_salaries(self):
        """Procesa información salarial"""
        self.df['salary_min'] = pd.to_numeric(self.df['salary_min'], errors='coerce')
        self.df['salary_max'] = pd.to_numeric(self.df['salary_max'], errors='coerce')
        self.df['salary_avg'] = (self.df['salary_min'] + self.df['salary_max']) / 2
        
        # Filtrar salarios irrealistas
        self.df.loc[(self.df['salary_avg'] < 18000) | (self.df['salary_avg'] > 150000), 'salary_avg'] = np.nan
        
        valid_salaries = self.df['salary_avg'].notna().sum()
        print(f"💰 {valid_salaries} ofertas con salario válido ({valid_salaries/len(self.df)*100:.1f}%)")
        
        if valid_salaries > 0:
            print(f"💶 Salario promedio: {self.df['salary_avg'].mean():,.0f}€")
        
        return self
    
    def process_dates(self):
        """Procesa fechas"""
        self.df['created'] = pd.to_datetime(self.df['created'], errors='coerce')
        self.df['collected_at'] = pd.to_datetime(self.df['collected_at'], errors='coerce')
        
        self.df['created_year'] = self.df['created'].dt.year
        self.df['created_month'] = self.df['created'].dt.month
        self.df['created_week'] = self.df['created'].dt.isocalendar().week
        
        print(f"📅 Rango: {self.df['created'].min()} → {self.df['created'].max()}")
        return self
    
    def clean_company_names(self):
        """Limpia nombres de empresas"""
        def clean_company(company):
            if pd.isna(company):
                return 'No especificada'
            
            company = str(company).strip()
            suffixes = [' S.L.', ' S.A.', ' SL', ' SA', ' Ltd', ' Inc', ' Corp']
            
            for suffix in suffixes:
                company = company.replace(suffix, '')
            
            return company.strip()
        
        self.df['company_clean'] = self.df['company_name'].apply(clean_company)
        print(f"🏢 {self.df['company_clean'].nunique()} empresas únicas")
        return self
    
    def extract_skills(self):
        """Extrae skills de las descripciones"""
        def get_skills(description):
            if pd.isna(description):
                return []
            
            description_lower = str(description).lower()
            found_skills = []
            
            for skill, pattern in self.SKILLS.items():
                if re.search(pattern, description_lower, re.IGNORECASE):
                    found_skills.append(skill)
            
            return found_skills
        
        print("🔍 Extrayendo skills...")
        self.df['skills'] = self.df['description'].apply(get_skills)
        self.df['num_skills'] = self.df['skills'].apply(len)
        
        print(f"✅ Promedio: {self.df['num_skills'].mean():.1f} skills por oferta")
        
        # Top skills
        all_skills = [skill for skills_list in self.df['skills'] for skill in skills_list]
        skill_counts = Counter(all_skills)
        
        print("\n🔥 Top 10 skills:")
        for skill, count in skill_counts.most_common(10):
            percentage = (count / len(self.df)) * 100
            print(f"   {skill:15} {count:4} ({percentage:5.1f}%)")
        
        return self
    
    def detect_ai_jobs(self):
        """Detecta ofertas relacionadas con IA/ML"""
        def has_ai(description):
            if pd.isna(description):
                return False
            
            description_lower = str(description).lower()
            return any(keyword in description_lower for keyword in self.AI_KEYWORDS)
        
        self.df['is_ai_related'] = self.df['description'].apply(has_ai)
        ai_count = self.df['is_ai_related'].sum()
        
        print(f"🤖 {ai_count} ofertas de IA/ML ({ai_count/len(self.df)*100:.1f}%)")
        return self
    
    def save_cleaned_data(self, output_file=None):
        """Guarda los datos limpios"""
        if output_file is None:
            output_file = PROCESSED_DATA_DIR / "jobs_cleaned.csv"
        
        columns_to_keep = [
            'id', 'title', 'company_clean', 'city', 'role_category', 'seniority',
            'salary_min', 'salary_max', 'salary_avg', 'description',
            'created', 'skills', 'num_skills', 'is_ai_related',
            'redirect_url', 'contract_type', 'contract_time'
        ]
        
        df_clean = self.df[columns_to_keep].copy()
        df_clean = df_clean.rename(columns={
            'company_clean': 'company',
            'redirect_url': 'url'
        })
        
        df_clean.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Datos guardados: {output_file}")
        print(f"📊 {len(df_clean)} ofertas × {len(df_clean.columns)} columnas")
        
        return df_clean
    
    def process_all(self):
        """Ejecuta todo el pipeline de limpieza"""
        print("🚀 Iniciando limpieza de datos...")
        print("=" * 70)
        
        self.load_data()
        self.remove_duplicates()
        self.clean_locations()
        self.extract_seniority()
        self.categorize_roles()
        self.process_salaries()
        self.process_dates()
        self.clean_company_names()
        self.extract_skills()
        self.detect_ai_jobs()
        
        print("\n" + "=" * 70)
        df_clean = self.save_cleaned_data()
        
        print("\n✅ LIMPIEZA COMPLETADA")
        print("=" * 70)
        print(f"📊 Resumen final:")
        print(f"   • {len(df_clean)} ofertas únicas")
        print(f"   • {df_clean['city'].nunique()} ciudades")
        print(f"   • {df_clean['company'].nunique()} empresas")
        print(f"   • {df_clean['salary_avg'].notna().sum()} con salario")
        print(f"   • {df_clean['is_ai_related'].sum()} ofertas IA/ML")
        
        return df_clean


def main():
    """Función principal"""
    processor = DataProcessor()
    df_clean = processor.process_all()
    return df_clean


if __name__ == "__main__":
    df = main()