"""
Script para generar todas las visualizaciones del proyecto
Guarda las imágenes en /images para el README
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import Counter
import ast

# Crear carpeta de imágenes
IMAGES_DIR = Path("../images")
IMAGES_DIR.mkdir(exist_ok=True)

class JobMarketVisualizer:
    """
    Clase para generar todas las visualizaciones del proyecto
    """
    
    def __init__(self, data_path):
        """Carga los datos"""
        self.df = pd.read_csv(data_path)
        self.df['skills'] = self.df['skills'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else []
        )
        self.df['created'] = pd.to_datetime(self.df['created'])
        
        print(f"📊 {len(self.df)} ofertas cargadas")
    
    def plot_role_distribution(self):
        """Gráfico de distribución de roles"""
        role_counts = self.df['role_category'].value_counts()
        
        fig = px.bar(
            x=role_counts.values,
            y=role_counts.index,
            orientation='h',
            title="Distribución de Roles de Data Science en España",
            labels={'x': 'Número de Ofertas', 'y': ''},
            color=role_counts.values,
            color_continuous_scale='Blues',
            text=role_counts.values
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=500,
            font=dict(size=14)
        )
        
        fig.write_image(IMAGES_DIR / "role_distribution.png", width=1200, height=600)
        print("✅ role_distribution.png guardado")
        return fig
    
    def plot_top_cities(self):
        """Gráfico de top ciudades"""
        top_cities = self.df['city'].value_counts().head(10)
        
        fig = px.bar(
            x=top_cities.index,
            y=top_cities.values,
            title="Top 10 Ciudades con Más Ofertas de Data Science",
            labels={'x': 'Ciudad', 'y': 'Número de Ofertas'},
            color=top_cities.values,
            color_continuous_scale='Viridis',
            text=top_cities.values
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            height=500
        )
        
        fig.write_image(IMAGES_DIR / "top_cities.png", width=1200, height=600)
        print("✅ top_cities.png guardado")
        return fig
    
    def plot_top_skills(self):
        """Gráfico de top skills"""
        all_skills = [skill for skills_list in self.df['skills'] 
                     if skills_list for skill in skills_list]
        skill_counts = Counter(all_skills)
        top_skills = dict(skill_counts.most_common(15))
        
        fig = px.bar(
            x=list(top_skills.values()),
            y=list(top_skills.keys()),
            orientation='h',
            title="Top 15 Skills Más Demandadas",
            labels={'x': 'Número de Ofertas', 'y': ''},
            color=list(top_skills.values()),
            color_continuous_scale='Reds',
            text=list(top_skills.values())
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=600,
            font=dict(size=14)
        )
        
        fig.write_image(IMAGES_DIR / "top_skills.png", width=1200, height=700)
        print("✅ top_skills.png guardado")
        return fig
    
    def plot_salary_distribution(self):
        """Distribución de salarios"""
        df_salary = self.df[self.df['salary_avg'].notna()].copy()
        
        if len(df_salary) < 10:
            print("⚠️  Pocos datos salariales, saltando visualización")
            return None
        
        fig = px.histogram(
            df_salary,
            x='salary_avg',
            nbins=25,
            title=f"Distribución de Salarios ({len(df_salary)} ofertas)",
            labels={'salary_avg': 'Salario Anual (€)', 'count': 'Ofertas'},
            color_discrete_sequence=['#3498db']
        )
        
        mean_sal = df_salary['salary_avg'].mean()
        median_sal = df_salary['salary_avg'].median()
        
        fig.add_vline(x=mean_sal, line_dash="dash", line_color="red",
                     annotation_text=f"Media: {mean_sal:,.0f}€")
        fig.add_vline(x=median_sal, line_dash="dash", line_color="green",
                     annotation_text=f"Mediana: {median_sal:,.0f}€")
        
        fig.update_layout(height=500)
        
        fig.write_image(IMAGES_DIR / "salary_distribution.png", width=1200, height=600)
        print("✅ salary_distribution.png guardado")
        return fig
    
    def plot_salary_by_role(self):
        """Salarios por rol"""
        df_salary = self.df[self.df['salary_avg'].notna()].copy()
        
        if len(df_salary) < 10:
            return None
        
        salary_by_role = df_salary.groupby('role_category')['salary_avg'].agg(['mean', 'count'])
        salary_by_role = salary_by_role[salary_by_role['count'] >= 5].sort_values('mean', ascending=False)
        
        fig = px.bar(
            x=salary_by_role.index,
            y=salary_by_role['mean'],
            title="Salarios Promedio por Tipo de Rol",
            labels={'x': '', 'y': 'Salario Anual (€)'},
            color=salary_by_role['mean'],
            color_continuous_scale='Greens',
            text=salary_by_role['mean'].apply(lambda x: f"{x:,.0f}€")
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=500, xaxis_tickangle=-45)
        
        fig.write_image(IMAGES_DIR / "salary_by_role.png", width=1200, height=600)
        print("✅ salary_by_role.png guardado")
        return fig
    
    def plot_ai_comparison(self):
        """Comparación IA vs No-IA"""
        ai_stats = self.df.groupby('is_ai_related').agg({
            'id': 'count',
            'salary_avg': 'mean'
        })
        
        ai_stats.index = ['Otros roles', 'IA/ML']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Distribución de Ofertas", "Salario Promedio"),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=ai_stats.index,
                values=ai_stats['id'],
                marker_colors=['#95a5a6', '#e74c3c'],
                textinfo='label+percent'
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=ai_stats.index,
                y=ai_stats['salary_avg'],
                marker_color=['#95a5a6', '#e74c3c'],
                text=ai_stats['salary_avg'].apply(lambda x: f"{x:,.0f}€" if pd.notna(x) else "N/A"),
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Comparación: Ofertas de IA/ML vs Resto",
            height=400,
            showlegend=False
        )
        
        fig.write_image(IMAGES_DIR / "ai_comparison.png", width=1200, height=500)
        print("✅ ai_comparison.png guardado")
        return fig
    
    def plot_temporal_trend(self):
        """Tendencia temporal"""
        df_temporal = self.df.set_index('created').resample('M').size()
        
        fig = px.line(
            x=df_temporal.index,
            y=df_temporal.values,
            title="Evolución Temporal de Ofertas de Trabajo",
            labels={'x': 'Fecha', 'y': 'Número de Ofertas'},
            markers=True
        )
        
        fig.update_traces(line_color='#3498db', line_width=3)
        fig.update_layout(height=500)
        
        fig.write_image(IMAGES_DIR / "temporal_trend.png", width=1200, height=600)
        print("✅ temporal_trend.png guardado")
        return fig
    
    def generate_all(self):
        """Genera todas las visualizaciones"""
        print("\n🎨 Generando visualizaciones...")
        print("=" * 60)
        
        self.plot_role_distribution()
        self.plot_top_cities()
        self.plot_top_skills()
        self.plot_salary_distribution()
        self.plot_salary_by_role()
        self.plot_ai_comparison()
        self.plot_temporal_trend()
        
        print("\n" + "=" * 60)
        print(f"✅ Todas las imágenes guardadas en: {IMAGES_DIR}")
        print("🎯 Listas para incluir en el README")


def main():
    """Función principal"""
    # Buscar el archivo en múltiples ubicaciones posibles
    possible_paths = [
        Path("../data/processed/jobs_cleaned.csv"),
        Path("data/processed/jobs_cleaned.csv"),
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
        print("\nEjecuta primero: python src/data_processing.py")
        return
    
    print(f"📂 Usando: {data_path}")
    visualizer = JobMarketVisualizer(data_path)
    visualizer.generate_all()


if __name__ == "__main__":
    main()