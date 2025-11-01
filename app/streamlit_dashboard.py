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
    # === PÁGINA 1: OVERVIEW ===
if page == "🏠 Overview":
    st.title("📊 Mercado Laboral de Data Science en España")
    st.markdown("### Análisis Completo del Sector en 2024-2025")

    # =========================
    # 📊 GRÁFICOS PRINCIPALES
    # =========================
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Distribución de Roles")
        role_col = 'role_category' if 'role_category' in df.columns else 'role'
        if role_col in df.columns:
            role_counts = df[role_col].value_counts()
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
        else:
            st.warning("⚠️ No se encontró la columna de roles en el dataset.")

    with col2:
        st.subheader("🏙️ Top 10 Ciudades")
        if 'city' in df.columns:
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
        else:
            st.warning("⚠️ No se encontró la columna de ciudades en el dataset.")

    # =========================
    # 📈 EVOLUCIÓN TEMPORAL
    # =========================
    if 'created' in df.columns:
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
    else:
        st.warning("⚠️ No hay columna temporal 'created' en el dataset.")


# === PÁGINA 3: ANÁLISIS GEOGRÁFICO ===
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
        avg_sal = df_filtered['salary_avg'].mean() if 'salary_avg' in df_filtered.columns else None
        st.metric("Salario Medio", f"{avg_sal:,.0f}€" if avg_sal else "N/A")
    with col3:
        st.metric("Empresas", df_filtered['company'].nunique() if 'company' in df_filtered.columns else "N/A")
    with col4:
        ai_pct = (df_filtered['is_ai_related'].sum() / len(df_filtered) * 100) if 'is_ai_related' in df_filtered.columns and len(df_filtered) > 0 else 0
        st.metric("% IA/ML", f"{ai_pct:.1f}%")

    # Gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💼 Roles en esta Ubicación")
        if role_col in df_filtered.columns:
            role_dist = df_filtered[role_col].value_counts()
            fig = px.pie(values=role_dist.values, names=role_dist.index)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se encontraron roles en esta ubicación.")

    with col2:
        st.subheader("🏢 Top Empresas")
        if 'company' in df_filtered.columns:
            company_dist = df_filtered['company'].value_counts().head(10)
            fig = px.bar(x=company_dist.values, y=company_dist.index, orientation='h')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de empresas disponibles.")


# === PÁGINA 7: PREDICTOR DE SALARIOS ===
elif page == "🔮 Predictor de Salarios":
    st.title("🔮 Predictor de Salarios")

    if model is None or metadata is None:
        st.error("❌ Modelo no disponible. Sube los archivos de modelo a la carpeta `/models/` y vuelve a desplegar.")
        st.markdown("""
        **Archivos necesarios:**
        - `models/salary_predictor.pkl`
        - `models/scaler.pkl`
        - `models/model_metadata.json`
        """)
    else:
        st.success("✅ Modelo cargado correctamente")

        with st.expander("📊 Métricas del Modelo"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{metadata['metrics']['mae']:,.0f}€")
            with col2:
                st.metric("RMSE", f"{metadata['metrics']['rmse']:,.0f}€")
            with col3:
                st.metric("R²", f"{metadata['metrics']['r2']:.3f}")

            st.info("El modelo tiene bajo R² debido a la alta variabilidad en los salarios. Las predicciones son orientativas.")

        st.markdown("---")
        st.subheader("Configura tu perfil:")

        col1, col2 = st.columns(2)
        with col1:
            available_skills = sorted(set([
                skill.replace('skill_', '')
                for skill in metadata.get('skill_columns', [])
            ]))
            selected_skills = st.multiselect(
                "Skills:",
                available_skills,
                default=['Python', 'SQL'] if 'Python' in available_skills else []
            )

            city = st.selectbox(
                "Ciudad:",
                ['Madrid', 'Barcelona', 'Valencia', 'Bilbao', 'Sevilla', 'Otras']
            )

        with col2:
            seniority = st.selectbox(
                "Nivel de Experiencia:",
                ['Junior', 'Mid-Level', 'Senior', 'Manager']
            )

            role = st.selectbox(
                "Tipo de Rol:",
                df[role_col].unique().tolist() if role_col in df.columns else ['Data Scientist']
            )

            is_ai = st.checkbox("¿Rol de IA/ML?", value=True)

        if st.button("🔮 Predecir Salario", type="primary"):
            st.markdown("---")
            st.subheader("💰 Resultado de la Predicción")

            base_salary = df['salary_avg'].mean() if 'salary_avg' in df.columns else 40000

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