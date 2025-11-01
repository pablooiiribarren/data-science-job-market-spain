"""
Generador de visualizaciones usando matplotlib/seaborn
Versión estable para Windows (sin kaleido)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import ast
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Detectar carpeta raíz
if Path("data").exists():
    BASE_DIR = Path(".")
elif Path("../data").exists():
    BASE_DIR = Path("..")
else:
    BASE_DIR = Path(".")

IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)


class JobMarketVisualizer:
    """Generador de visualizaciones con matplotlib"""
    
    def __init__(self, data_path):
        """Carga los datos"""
        self.df = pd.read_csv(data_path)
        self.df['skills'] = self.df['skills'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else []
        )
        self.df['created'] = pd.to_datetime(self.df['created'])
        print(f"📊 {len(self.df)} ofertas cargadas")
    
    def plot_role_distribution(self):
        """Distribución de roles"""
        role_counts = self.df['role_category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(role_counts)), role_counts.values, color=sns.color_palette("Blues_r", len(role_counts)))
        ax.set_yticks(range(len(role_counts)))
        ax.set_yticklabels(role_counts.index)
        ax.set_xlabel('Número de Ofertas', fontsize=14, fontweight='bold')
        ax.set_title('Distribución de Roles de Data Science en España', fontsize=16, fontweight='bold', pad=20)
        
        # Añadir valores en las barras
        for i, (bar, value) in enumerate(zip(bars, role_counts.values)):
            ax.text(value + 10, i, str(value), va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "role_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ role_distribution.png")
    
    def plot_top_cities(self):
        """Top ciudades"""
        top_cities = self.df['city'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = sns.color_palette("viridis", len(top_cities))
        bars = ax.bar(range(len(top_cities)), top_cities.values, color=colors)
        ax.set_xticks(range(len(top_cities)))
        ax.set_xticklabels(top_cities.index, rotation=45, ha='right')
        ax.set_ylabel('Número de Ofertas', fontsize=14, fontweight='bold')
        ax.set_title('Top 10 Ciudades con Más Ofertas de Data Science', fontsize=16, fontweight='bold', pad=20)
        
        # Valores en barras
        for bar, value in zip(bars, top_cities.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "top_cities.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ top_cities.png")
    
    def plot_top_skills(self):
        """Top skills"""
        all_skills = [skill for skills_list in self.df['skills'] if skills_list for skill in skills_list]
        skill_counts = Counter(all_skills)
        top_skills = dict(skill_counts.most_common(15))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette("Reds_r", len(top_skills))
        bars = ax.barh(range(len(top_skills)), list(top_skills.values()), color=colors)
        ax.set_yticks(range(len(top_skills)))
        ax.set_yticklabels(list(top_skills.keys()))
        ax.set_xlabel('Número de Ofertas', fontsize=14, fontweight='bold')
        ax.set_title('Top 15 Skills Más Demandadas', fontsize=16, fontweight='bold', pad=20)
        
        for i, (bar, value) in enumerate(zip(bars, top_skills.values())):
            ax.text(value + 2, i, str(value), va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "top_skills.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ top_skills.png")
    
    def plot_salary_distribution(self):
        """Distribución de salarios"""
        df_salary = self.df[self.df['salary_avg'].notna()].copy()
        
        if len(df_salary) < 10:
            print("⚠️  Pocos datos salariales, saltando")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(df_salary['salary_avg'], bins=25, color='#3498db', edgecolor='black', alpha=0.7)
        
        mean_sal = df_salary['salary_avg'].mean()
        median_sal = df_salary['salary_avg'].median()
        
        ax.axvline(mean_sal, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_sal:,.0f}€')
        ax.axvline(median_sal, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_sal:,.0f}€')
        
        ax.set_xlabel('Salario Anual (€)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Número de Ofertas', fontsize=14, fontweight='bold')
        ax.set_title(f'Distribución de Salarios ({len(df_salary)} ofertas)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "salary_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ salary_distribution.png")
    
    def plot_salary_by_role(self):
        """Salarios por rol"""
        df_salary = self.df[self.df['salary_avg'].notna()].copy()
        
        if len(df_salary) < 10:
            return
        
        salary_by_role = df_salary.groupby('role_category')['salary_avg'].agg(['mean', 'count'])
        salary_by_role = salary_by_role[salary_by_role['count'] >= 5].sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = sns.color_palette("Greens_r", len(salary_by_role))
        bars = ax.bar(range(len(salary_by_role)), salary_by_role['mean'], color=colors)
        ax.set_xticks(range(len(salary_by_role)))
        ax.set_xticklabels(salary_by_role.index, rotation=45, ha='right')
        ax.set_ylabel('Salario Anual (€)', fontsize=14, fontweight='bold')
        ax.set_title('Salarios Promedio por Tipo de Rol', fontsize=16, fontweight='bold', pad=20)
        
        for bar, value in zip(bars, salary_by_role['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:,.0f}€', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "salary_by_role.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ salary_by_role.png")
    
    def plot_ai_comparison(self):
        """Comparación IA vs No-IA"""
        ai_stats = self.df.groupby('is_ai_related').agg({
            'id': 'count',
            'salary_avg': 'mean'
        })
        ai_stats.index = ['Otros roles', 'IA/ML']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors_pie = ['#95a5a6', '#e74c3c']
        explode = (0, 0.1)
        ax1.pie(ai_stats['id'], labels=ai_stats.index, autopct='%1.1f%%',
               colors=colors_pie, explode=explode, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax1.set_title('Distribución de Ofertas', fontsize=14, fontweight='bold', pad=20)
        
        # Bar chart
        bars = ax2.bar(ai_stats.index, ai_stats['salary_avg'], color=colors_pie)
        ax2.set_ylabel('Salario Promedio (€)', fontsize=12, fontweight='bold')
        ax2.set_title('Salario Promedio', fontsize=14, fontweight='bold', pad=20)
        
        for bar, value in zip(bars, ai_stats['salary_avg']):
            if pd.notna(value):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:,.0f}€', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Comparación: Ofertas de IA/ML vs Resto', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "ai_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ ai_comparison.png")
    
    def plot_temporal_trend(self):
        """Tendencia temporal"""
        df_temporal = self.df.set_index('created').resample('M').size()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_temporal.index, df_temporal.values, marker='o', linewidth=3, color='#3498db', markersize=6)
        ax.fill_between(df_temporal.index, df_temporal.values, alpha=0.3, color='#3498db')
        
        ax.set_xlabel('Fecha', fontsize=14, fontweight='bold')
        ax.set_ylabel('Número de Ofertas', fontsize=14, fontweight='bold')
        ax.set_title('Evolución Temporal de Ofertas de Trabajo', fontsize=16, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "temporal_trend.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ temporal_trend.png")
    
    def plot_seniority_distribution(self):
        """Distribución por nivel de experiencia"""
        seniority_counts = self.df['seniority'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("Set2", len(seniority_counts))
        wedges, texts, autotexts = ax.pie(seniority_counts.values, labels=seniority_counts.index, 
                                           autopct='%1.1f%%', colors=colors, startangle=90,
                                           textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_title('Distribución por Nivel de Experiencia', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "seniority_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ seniority_distribution.png")
    
    def generate_summary_stats(self):
        """Genera archivo de texto con estadísticas"""
        stats_file = IMAGES_DIR / "summary_stats.txt"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RESUMEN ESTADÍSTICO - MERCADO LABORAL DATA SCIENCE ESPAÑA\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"📊 DATOS GENERALES:\n")
            f.write(f"   • Total ofertas: {len(self.df)}\n")
            f.write(f"   • Ciudades: {self.df['city'].nunique()}\n")
            f.write(f"   • Empresas: {self.df['company'].nunique()}\n")
            f.write(f"   • Ofertas con salario: {self.df['salary_avg'].notna().sum()} ({self.df['salary_avg'].notna().sum()/len(self.df)*100:.1f}%)\n")
            
            if self.df['salary_avg'].notna().sum() > 0:
                f.write(f"\n💰 SALARIOS:\n")
                f.write(f"   • Promedio: {self.df['salary_avg'].mean():,.0f}€\n")
                f.write(f"   • Mediana: {self.df['salary_avg'].median():,.0f}€\n")
                f.write(f"   • Mínimo: {self.df['salary_avg'].min():,.0f}€\n")
                f.write(f"   • Máximo: {self.df['salary_avg'].max():,.0f}€\n")
            
            f.write(f"\n💼 TOP 5 ROLES:\n")
            for i, (role, count) in enumerate(self.df['role_category'].value_counts().head(5).items(), 1):
                f.write(f"   {i}. {role}: {count} ({count/len(self.df)*100:.1f}%)\n")
            
            f.write(f"\n🏙️ TOP 5 CIUDADES:\n")
            for i, (city, count) in enumerate(self.df['city'].value_counts().head(5).items(), 1):
                f.write(f"   {i}. {city}: {count}\n")
            
            all_skills = [skill for skills_list in self.df['skills'] if skills_list for skill in skills_list]
            skill_counts = Counter(all_skills)
            
            f.write(f"\n🔥 TOP 10 SKILLS:\n")
            for i, (skill, count) in enumerate(skill_counts.most_common(10), 1):
                f.write(f"   {i}. {skill}: {count} ({count/len(self.df)*100:.1f}%)\n")
            
            f.write(f"\n🤖 IA/ML:\n")
            ai_count = self.df['is_ai_related'].sum()
            f.write(f"   • Ofertas IA/ML: {ai_count} ({ai_count/len(self.df)*100:.1f}%)\n")
        
        print("✅ summary_stats.txt")
    
    def generate_all(self):
        """Genera todas las visualizaciones"""
        print("\n🎨 Generando visualizaciones...")
        print("=" * 60)
        
        try:
            self.plot_role_distribution()
            self.plot_top_cities()
            self.plot_top_skills()
            self.plot_salary_distribution()
            self.plot_salary_by_role()
            self.plot_ai_comparison()
            self.plot_temporal_trend()
            self.plot_seniority_distribution()
            self.generate_summary_stats()
            
            print("\n" + "=" * 60)
            print(f"✅ Todas las imágenes guardadas en: {IMAGES_DIR.absolute()}")
            print("🎯 Listas para incluir en el README")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Función principal"""
    possible_paths = [
        Path("data/processed/jobs_cleaned.csv"),
        Path("../data/processed/jobs_cleaned.csv"),
        Path("./data/processed/jobs_cleaned.csv")
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print("❌ Error: No se encuentra jobs_cleaned.csv")
        print("\nBuscado en:")
        for path in possible_paths:
            print(f"   • {path.absolute()}")
        return
    
    print(f"📂 Usando: {data_path}")
    visualizer = JobMarketVisualizer(data_path)
    visualizer.generate_all()


if __name__ == "__main__":
    main()