"""
Extractor mejorado de skills con variaciones y sinónimos
"""

import pandas as pd
import re
from collections import Counter
from pathlib import Path
from config import PROCESSED_DATA_DIR

class ImprovedSkillsExtractor:
    """
    Extractor mejorado que busca variaciones y contexto de skills
    """
    
    def __init__(self):
        # Skills con múltiples variaciones
        self.SKILLS_PATTERNS = {
            # Lenguajes de programación
            'Python': [
                r'\bpython\b', r'\bpython\s*[23]', r'\bpyspark\b', 
                r'\bpandas\b', r'\bnumpy\b', r'\bscipy\b'
            ],
            'R': [
                r'\b r\b', r'\br\s+programming\b', r'\brstudio\b',
                r'\btidyverse\b', r'\bggplot\b', r'\bshiny\b'
            ],
            'SQL': [
                r'\bsql\b', r'\bmysql\b', r'\bpostgresql\b', r'\bpostgres\b',
                r'\bt-sql\b', r'\bpl/sql\b', r'\bnosql\b'
            ],
            'Java': [r'\bjava\b', r'\bjava\s*[0-9]', r'\bjavascript\b'],
            'Scala': [r'\bscala\b'],
            'Julia': [r'\bjulia\b'],
            
            # Frameworks ML/DL
            'TensorFlow': [r'\btensorflow\b', r'\btf\b', r'\bkeras\b'],
            'PyTorch': [r'\bpytorch\b', r'\btorch\b'],
            'scikit-learn': [r'\bscikit.learn\b', r'\bsklearn\b', r'\bsci-kit\b'],
            'XGBoost': [r'\bxgboost\b', r'\bxgb\b'],
            'LightGBM': [r'\blightgbm\b', r'\blgbm\b'],
            
            # Big Data
            'Spark': [r'\bspark\b', r'\bpyspark\b', r'\bapache spark\b'],
            'Hadoop': [r'\bhadoop\b', r'\bhdfs\b', r'\bmapreduce\b'],
            'Kafka': [r'\bkafka\b', r'\bapache kafka\b'],
            'Airflow': [r'\bairflow\b', r'\bapache airflow\b'],
            'Databricks': [r'\bdatabricks\b'],
            
            # Bases de datos
            'PostgreSQL': [r'\bpostgresql\b', r'\bpostgres\b'],
            'MySQL': [r'\bmysql\b'],
            'MongoDB': [r'\bmongodb\b', r'\bmongo\b'],
            'Redis': [r'\bredis\b'],
            'Cassandra': [r'\bcassandra\b'],
            'Elasticsearch': [r'\belasticsearch\b', r'\belastic\b'],
            'Oracle': [r'\boracle\b', r'\boracle db\b'],
            'SQL Server': [r'\bsql server\b', r'\bmssql\b'],
            
            # Cloud
            'AWS': [
                r'\baws\b', r'\bamazon web services\b', r'\bs3\b',
                r'\bec2\b', r'\blambda\b', r'\bredshift\b', r'\bsagemaker\b'
            ],
            'Azure': [
                r'\bazure\b', r'\bmicrosoft azure\b', r'\bazure ml\b',
                r'\bazure databricks\b'
            ],
            'GCP': [
                r'\bgcp\b', r'\bgoogle cloud\b', r'\bbigquery\b',
                r'\bvertex ai\b'
            ],
            
            # BI Tools
            'Tableau': [r'\btableau\b'],
            'Power BI': [r'\bpower bi\b', r'\bpowerbi\b', r'\bpower-bi\b'],
            'Looker': [r'\blooker\b'],
            'QlikView': [r'\bqlikview\b', r'\bqlik\b'],
            
            # DevOps & Tools
            'Docker': [r'\bdocker\b', r'\bcontainer\b'],
            'Kubernetes': [r'\bkubernetes\b', r'\bk8s\b'],
            'Git': [r'\bgit\b', r'\bgithub\b', r'\bgitlab\b'],
            'CI/CD': [r'\bci/cd\b', r'\bjenkins\b', r'\btravis\b'],
            
            # Metodologías y conceptos
            'Machine Learning': [
                r'\bmachine learning\b', r'\bml\b', r'\bsupervised\b',
                r'\bunsupervised\b', r'\breinforcement learning\b'
            ],
            'Deep Learning': [
                r'\bdeep learning\b', r'\bneural network\b', r'\bcnn\b',
                r'\brnn\b', r'\blstm\b', r'\btransformer\b'
            ],
            'NLP': [
                r'\bnlp\b', r'\bnatural language\b', r'\bbert\b',
                r'\bgpt\b', r'\bllm\b', r'\btext mining\b'
            ],
            'Computer Vision': [
                r'\bcomputer vision\b', r'\bcv\b', r'\bimage processing\b',
                r'\byolo\b', r'\bresnet\b'
            ],
            'Statistics': [
                r'\bstatistics\b', r'\bstatistical\b', r'\bhypothesis testing\b',
                r'\bregression\b', r'\banova\b'
            ],
            'ETL': [r'\betl\b', r'\bextract transform load\b'],
            'Data Visualization': [
                r'\bvisualization\b', r'\bdashboard\b', r'\bdata viz\b'
            ],
            'MLOps': [r'\bmlops\b', r'\bml ops\b', r'\bmodel deployment\b'],
            
            # Otros
            'Excel': [r'\bexcel\b', r'\bvba\b', r'\bspreadsheet\b'],
            'Linux': [r'\blinux\b', r'\bunix\b', r'\bbash\b'],
        }
    
    def extract_skills(self, description):
        """
        Extrae skills con patrones mejorados
        
        Args:
            description: Descripción de la oferta
            
        Returns:
            list: Lista de skills encontradas
        """
        if pd.isna(description):
            return []
        
        description_lower = str(description).lower()
        found_skills = []
        
        for skill, patterns in self.SKILLS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, description_lower, re.IGNORECASE):
                    found_skills.append(skill)
                    break  # Solo añadir una vez por skill
        
        return list(set(found_skills))  # Eliminar duplicados
    
    def process_dataset(self, input_file=None):
        """
        Procesa el dataset completo con el nuevo extractor
        
        Args:
            input_file: Ruta al CSV limpio
        """
        if input_file is None:
            input_file = PROCESSED_DATA_DIR / "jobs_cleaned.csv"
        
        print("📂 Cargando dataset limpio...")
        df = pd.read_csv(input_file)
        print(f"✅ {len(df)} ofertas cargadas\n")
        
        print("🔍 Re-extrayendo skills con patrones mejorados...")
        
        # Convertir string de lista a lista real si es necesario
        if df['skills'].dtype == 'object' and isinstance(df['skills'].iloc[0], str):
            import ast
            df['skills'] = df['skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        
        # Re-extraer skills
        df['skills_improved'] = df['description'].apply(self.extract_skills)
        df['num_skills_improved'] = df['skills_improved'].apply(len)
        
        # Estadísticas
        print(f"✅ Extracción completada\n")
        print("📊 Comparación:")
        print(f"   Método anterior: {df['num_skills'].mean():.2f} skills por oferta")
        print(f"   Método mejorado: {df['num_skills_improved'].mean():.2f} skills por oferta")
        print(f"   Mejora: {((df['num_skills_improved'].mean() / df['num_skills'].mean() - 1) * 100):.0f}% más skills detectadas\n")
        
        # Top skills con método mejorado
        all_skills = [skill for skills_list in df['skills_improved'] for skill in skills_list]
        skill_counts = Counter(all_skills)
        
        print("🔥 Top 20 skills más demandadas (mejorado):")
        print("-" * 60)
        for i, (skill, count) in enumerate(skill_counts.most_common(20), 1):
            percentage = (count / len(df)) * 100
            bar = "█" * int(percentage / 2)
            print(f"{i:2}. {skill:25} {count:4} ({percentage:5.1f}%) {bar}")
        
        # Reemplazar columnas antiguas
        df['skills'] = df['skills_improved']
        df['num_skills'] = df['num_skills_improved']
        df = df.drop(columns=['skills_improved', 'num_skills_improved'])
        
        # Guardar
        output_file = PROCESSED_DATA_DIR / "jobs_cleaned.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n💾 Dataset actualizado guardado: {output_file}")
        
        # Análisis adicional
        print("\n" + "="*60)
        print("📈 ANÁLISIS DE SKILLS")
        print("="*60)
        
        # Skills por categoría de rol
        print("\n💼 Skills promedio por tipo de rol:")
        role_skills = df.groupby('role_category')['num_skills'].mean().sort_values(ascending=False)
        for role, avg_skills in role_skills.items():
            print(f"   {role:25} {avg_skills:.1f} skills")
        
        # Skills por nivel de experiencia
        print("\n📊 Skills promedio por nivel de experiencia:")
        seniority_skills = df.groupby('seniority')['num_skills'].mean().sort_values(ascending=False)
        for level, avg_skills in seniority_skills.items():
            print(f"   {level:20} {avg_skills:.1f} skills")
        
        # Ofertas con más skills
        print("\n🏆 Top 5 ofertas con más skills requeridas:")
        top_skilled = df.nlargest(5, 'num_skills')[['title', 'company', 'city', 'num_skills', 'skills']]
        for idx, row in top_skilled.iterrows():
            print(f"\n   • {row['title'][:50]}")
            print(f"     {row['company']} - {row['city']}")
            print(f"     {row['num_skills']} skills: {', '.join(row['skills'][:10])}")
        
        return df


def main():
    """Función principal"""
    print("\n" + "🚀 IMPROVED SKILLS EXTRACTOR ".center(60, "="))
    print()
    
    extractor = ImprovedSkillsExtractor()
    df = extractor.process_dataset()
    
    print("\n" + "="*60)
    print("✅ Proceso completado")
    print("🎯 Siguiente paso: Análisis exploratorio visual")
    print("="*60)
    
    return df


if __name__ == "__main__":
    df = main()