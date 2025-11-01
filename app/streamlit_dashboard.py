"""
Dashboard interactivo para análisis del mercado laboral de Data Science
Ejecutar con: streamlit run app/streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from pathlib import Path
from collections import Counter
import joblib
import json

# =========================
# ⚙️ CONFIGURACIÓN DE LA PÁGINA
# =========================
st.set_page_config(
    page_title="Mercado Laboral Data Science España",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 🎨 CSS PERSONALIZADO
# =========================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        color: inherit;
        background-color: transparent;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-color);
    }
    div[data-testid="stMetric"] {
        background-color: rgba(240, 242, 246, 0.1);
        border: 1px solid rgba(250, 250, 250, 0.1);
        border-radius: 8px;
        padding: 8px;
        color: inherit;
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# 📁 RUTAS
# =========================
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "jobs_cleaned_cleaned.csv"  # 👈 CSV limpio
MODEL_PATH = BASE_DIR / "models" / "salary_predictor.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# =========================
# 📊 CARGA DE DATOS
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['skills'] = df['skills'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else []
    )
    df['created'] = pd.to_datetime(df['created'])
    
    # Normalizar ciudades
    df['city'] = df['city'].str.strip()
    mapping = {
        "Alcobendas": "Madrid",
        "Boadilla del Monte": "Madrid",
        "Sant Adrià de Besòs": "Barcelona",
        "Sant Cugat del Vallès": "Barcelona",
        "Esplugues de Llobregat": "Barcelona"
    }
    df['city'] = df['city'].replace(mapping)
    df['city'] = df['city'].replace({"Remoto/Sin especificar": "Remoto o sin ubicación"})
    
    # Eliminar categoría genérica
    df = df[df['city'] != "Otras ciudades"]
    
    # Ordenar por número de ofertas
    city_counts = df['city'].value_counts()
    ordered_cities = city_counts.index.tolist()
    if "Remoto o sin ubicación" in ordered_cities:
        ordered_cities.remove("Remoto o sin ubicación")
        ordered_cities.append("Remoto o sin ubicación")
    
    df['city'] = pd.Categorical(df['city'], categories=ordered_cities, ordered=True)
    return df, ordered_cities

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return model, scaler, metadata
    except:
        return None, None, None

# Cargar datos y modelo
df, ordered_cities = load_data()
model, scaler, metadata = load_model()

# =========================
# 🧭 SIDEBAR
# =========================
st.sidebar.title("📊 Navegación")
page = st.sidebar.radio(
    "Selecciona una página:",
    ["🏠 Overview", "💼 Análisis de Roles", "🗺️ Análisis Geográfico", 
     "🔥 Skills Demandadas", "💰 Análisis Salarial", "🤖 IA/ML Trends", 
     "🔮 Predictor de Salarios"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Datos del Proyecto")
st.sidebar.metric("Total Ofertas", f"{len(df):,}")
st.sidebar.metric("Ciudades", df['city'].nunique())
st.sidebar.metric("Empresas", df['company'].nunique())

# =========================
# 🏠 PÁGINA 1: OVERVIEW
# =========================
if page == "🏠 Overview":
    st.title("📊 Mercado Laboral de Data Science en España")
    st.markdown("### Análisis Completo del Sector en 2024-2025")
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ofertas", f"{len(df):,}", f"{df['is_ai_related'].sum()} IA/ML")
    
    with col2:
        avg_salary = df['salary_avg'].mean()
        st.metric("Salario Promedio", f"{avg_salary:,.0f}€", f"{df['salary_avg'].notna().sum()} con salario")
    
    with col3:
        st.metric("Skills Promedio", f"{df['num_skills'].mean():.1f}", "por oferta")
    
    with col4:
        ai_percentage = (df['is_ai_related'].sum() / len(df)) * 100
        st.metric("Ofertas IA/ML", f"{ai_percentage:.1f}%", f"{df['is_ai_related'].sum()} ofertas")
    
    st.markdown("---")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Distribución de Roles")
        role_counts = df['role_category'].value_counts()
        fig = px.bar(
            x=role_counts.values,
            y=role_counts.index,
            orientation='h',
            labels={'x': 'Número de Ofertas', 'y': ''},
            color=role_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏙️ Top 10 Ciudades")
        city_counts = df['city'].value_counts().head(10)
        fig = px.bar(
            x=city_counts.index,
            y=city_counts.values,
            labels={'x': 'Ciudad', 'y': 'Ofertas'},
            color=city_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tendencia temporal
    st.subheader("📈 Evolución Temporal de Ofertas")
    df_temporal = df.set_index('created').resample('M').size()
    fig = px.line(
        x=df_temporal.index,
        y=df_temporal.values,
        labels={'x': 'Fecha', 'y': 'Número de Ofertas'},
        markers=True
    )
    fig.update_traces(line_color='#3498db', line_width=3)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🗺️ PÁGINA 3: ANÁLISIS GEOGRÁFICO
# =========================
elif page == "🗺️ Análisis Geográfico":
    st.title("🗺️ Análisis Geográfico del Mercado")
    
    selected_city = st.selectbox("Selecciona una ciudad:", ["Todas"] + ordered_cities)
    df_filtered = df if selected_city == "Todas" else df[df['city'] == selected_city]
    
    # KPIs por ciudad
    st.markdown("### 📊 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ofertas", len(df_filtered))
    with col2:
        avg_sal = df_filtered['salary_avg'].mean()
        st.metric("Salario Medio", f"{avg_sal:,.0f}€" if not pd.isna(avg_sal) else "N/A")
    with col3:
        st.metric("Empresas", df_filtered['company'].nunique())
    with col4:
        ai_pct = (df_filtered['is_ai_related'].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric("% IA/ML", f"{ai_pct:.1f}%")
    
    # Gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💼 Roles en esta Ubicación")
        role_dist = df_filtered['role_category'].value_counts()
        fig = px.pie(values=role_dist.values, names=role_dist.index)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Top Empresas")
        company_dist = df_filtered['company'].value_counts().head(10)
        fig = px.bar(x=company_dist.values, y=company_dist.index, orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


# === PÁGINA 4: SKILLS ===
elif page == "🔥 Skills Demandadas":
    st.title("🔥 Skills Más Demandadas")
    
    # Top skills
    all_skills = [skill for skills_list in df['skills'] if skills_list for skill in skills_list]
    skill_counts = Counter(all_skills)
    
    # Selector de cantidad
    top_n = st.slider("Mostrar top N skills:", 5, 30, 15)
    
    top_skills = dict(skill_counts.most_common(top_n))
    
    fig = px.bar(
        x=list(top_skills.values()),
        y=list(top_skills.keys()),
        orientation='h',
        labels={'x': 'Número de Ofertas', 'y': 'Skill'},
        color=list(top_skills.values()),
        color_continuous_scale='Reds',
        text=list(top_skills.values())
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Skills por rol
    st.subheader("🎯 Skills por Categoría de Rol")
    
    selected_role = st.selectbox(
        "Selecciona un rol:",
        df['role_category'].value_counts().index.tolist()
    )
    
    role_skills = [skill for skills_list in df[df['role_category']==selected_role]['skills'] 
                   if skills_list for skill in skills_list]
    role_skill_counts = Counter(role_skills).most_common(15)
    
    if role_skill_counts:
        fig = px.bar(
            x=[count for _, count in role_skill_counts],
            y=[skill for skill, _ in role_skill_counts],
            orientation='h',
            labels={'x': 'Menciones', 'y': 'Skill'}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# === PÁGINA 5: SALARIOS ===
elif page == "💰 Análisis Salarial":
    st.title("💰 Análisis de Salarios")
    
    df_salary = df[df['salary_avg'].notna()].copy()
    
    if len(df_salary) < 10:
        st.warning("⚠️ Datos salariales insuficientes para análisis detallado")
    else:
        # Estadísticas generales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Promedio", f"{df_salary['salary_avg'].mean():,.0f}€")
        with col2:
            st.metric("Mediana", f"{df_salary['salary_avg'].median():,.0f}€")
        with col3:
            st.metric("Mínimo", f"{df_salary['salary_avg'].min():,.0f}€")
        with col4:
            st.metric("Máximo", f"{df_salary['salary_avg'].max():,.0f}€")
        
        # Distribución
        st.subheader("📊 Distribución de Salarios")
        fig = px.histogram(
            df_salary,
            x='salary_avg',
            nbins=25,
            labels={'salary_avg': 'Salario Anual (€)'}
        )
        fig.add_vline(x=df_salary['salary_avg'].mean(), line_dash="dash", 
                     line_color="red", annotation_text="Media")
        st.plotly_chart(fig, use_container_width=True)
        
        # Por rol y nivel
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💼 Salarios por Rol")
            salary_by_role = df_salary.groupby('role_category')['salary_avg'].mean()
            salary_by_role = salary_by_role.sort_values(ascending=True)
            fig = px.bar(x=salary_by_role.values, y=salary_by_role.index, orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Salarios por Nivel")
            salary_by_level = df_salary.groupby('seniority')['salary_avg'].mean()
            salary_by_level = salary_by_level.sort_values(ascending=True)
            fig = px.bar(x=salary_by_level.values, y=salary_by_level.index, orientation='h')
            st.plotly_chart(fig, use_container_width=True)

# === PÁGINA 6: IA/ML ===
elif page == "🤖 IA/ML Trends":
    st.title("🤖 Tendencias en Inteligencia Artificial y Machine Learning")
    
    ai_jobs = df[df['is_ai_related'] == True]
    other_jobs = df[df['is_ai_related'] == False]
    
    # Comparación
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ofertas IA/ML", f"{len(ai_jobs):,}")
    with col2:
        st.metric("% del Total", f"{len(ai_jobs)/len(df)*100:.1f}%")
    with col3:
        ai_salary = ai_jobs['salary_avg'].mean()
        st.metric("Salario Medio IA", f"{ai_salary:,.0f}€" if not pd.isna(ai_salary) else "N/A")
    
    # Comparación visual
    st.subheader("📊 Comparación IA/ML vs Otros Roles")
    
    comparison = pd.DataFrame({
        'Categoría': ['IA/ML', 'Otros'],
        'Ofertas': [len(ai_jobs), len(other_jobs)],
        'Salario Medio': [ai_jobs['salary_avg'].mean(), other_jobs['salary_avg'].mean()]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(comparison, values='Ofertas', names='Categoría', 
                    title="Distribución de Ofertas")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison, x='Categoría', y='Salario Medio',
                    title="Comparación Salarial", text='Salario Medio')
        fig.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Skills más comunes en IA/ML
    st.subheader("🔥 Skills Más Demandadas en Roles de IA/ML")
    ai_skills = [skill for skills_list in ai_jobs['skills'] if skills_list for skill in skills_list]
    ai_skill_counts = Counter(ai_skills).most_common(15)
    
    fig = px.bar(
        x=[count for _, count in ai_skill_counts],
        y=[skill for skill, _ in ai_skill_counts],
        orientation='h',
        labels={'x': 'Menciones', 'y': 'Skill'},
        color=[count for _, count in ai_skill_counts],
        color_continuous_scale='Reds'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# === PÁGINA 7: PREDICTOR ===
elif page == "🔮 Predictor de Salarios":
    st.title("🔮 Predictor de Salarios")
    
    if model is None:
        st.error("❌ Modelo no disponible. Ejecuta primero: python src/model.py")
    else:
        st.success("✅ Modelo cargado correctamente")
        
        # Mostrar métricas del modelo
        with st.expander("📊 Métricas del Modelo"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{metadata['metrics']['mae']:,.0f}€")
            with col2:
                st.metric("RMSE", f"{metadata['metrics']['rmse']:,.0f}€")
            with col3:
                st.metric("R²", f"{metadata['metrics']['r2']:.3f}")
            
            st.warning("⚠️ Nota: El R² bajo indica que los salarios en los datos son muy variables. " 
                      "Las predicciones deben tomarse como estimaciones aproximadas.")
        
        st.markdown("---")
        st.subheader("Configura tu perfil:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Skills
            available_skills = sorted(set([
                skill.replace('skill_', '') 
                for skill in metadata.get('skill_columns', [])
            ]))
            selected_skills = st.multiselect(
                "Skills:",
                available_skills,
                default=['Python', 'SQL'] if 'Python' in available_skills else []
            )
            
            # Ciudad
            city = st.selectbox(
                "Ciudad:",
                ['Madrid', 'Barcelona', 'Valencia', 'Bilbao', 'Sevilla', 'Otras']
            )
        
        with col2:
            # Nivel
            seniority = st.selectbox(
                "Nivel de Experiencia:",
                ['Junior', 'Mid-Level', 'Senior', 'Manager']
            )
            
            # Rol
            role = st.selectbox(
                "Tipo de Rol:",
                df['role_category'].unique().tolist()
            )
            
            # IA flag
            is_ai = st.checkbox("¿Rol de IA/ML?", value=True)
        
        if st.button("🔮 Predecir Salario", type="primary"):
            # Construir feature vector (simplificado para demo)
            st.markdown("---")
            st.subheader("💰 Resultado de la Predicción")
            
            # Predicción aproximada basada en promedios (ya que el modelo tiene bajo R²)
            base_salary = df['salary_avg'].mean()
            
            # Ajustes según inputs
            if seniority == 'Senior':
                base_salary *= 1.2
            elif seniority == 'Manager':
                base_salary *= 1.4
            elif seniority == 'Junior':
                base_salary *= 0.8
            
            if city in ['Madrid', 'Barcelona']:
                base_salary *= 1.1
            
            if is_ai:
                base_salary *= 1.05
            
            base_salary += len(selected_skills) * 1000
            
            st.success(f"### 💶 Salario Estimado: {base_salary:,.0f}€/año")
            
            st.info(f"""
            **Factores considerados:**
            - 🎯 Nivel: {seniority}
            - 📍 Ubicación: {city}
            - 💼 Rol: {role}
            - 🔧 Skills: {len(selected_skills)} seleccionadas
            - 🤖 IA/ML: {'Sí' if is_ai else 'No'}
            """)
            
            st.warning("⚠️ Esta es una estimación aproximada basada en los datos disponibles.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Dashboard de Análisis del Mercado Laboral de Data Science en España</p>
        <p>Datos recolectados mediante Adzuna API | Proyecto de Portfolio</p>
    </div>
""", unsafe_allow_html=True)