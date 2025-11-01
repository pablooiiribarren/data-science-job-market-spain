"""
Dashboard interactivo para análisis del mercado laboral de Data Science
Ejecutar con: streamlit run app/streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from pathlib import Path
import ast
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
.main { padding: 0rem 1rem; }
div[data-testid="stMetric"] {
    background-color: rgba(240, 242, 246, 0.1);
    border: 1px solid rgba(250, 250, 250, 0.1);
    border-radius: 8px;
    padding: 8px;
    color: inherit;
}
h1, h2, h3, h4, h5, h6 { color: var(--text-color); }
</style>
""", unsafe_allow_html=True)

# =========================
# 📁 RUTAS
# =========================
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "jobs_cleaned_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "salary_predictor.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# =========================
# 📊 CARGA DE DATOS
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['skills'] = df['skills'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else [])
    df['created'] = pd.to_datetime(df['created'], errors='coerce')

    # Limpieza y agrupaciones de ciudades
    mapping = {
        "Alcobendas": "Madrid",
        "Boadilla del Monte": "Madrid",
        "Sant Adrià de Besòs": "Barcelona",
        "Sant Cugat del Vallès": "Barcelona",
        "Esplugues de Llobregat": "Barcelona"
    }
    df['city'] = df['city'].replace(mapping)
    df['city'] = df['city'].replace({"Remoto/Sin especificar": "Remoto o sin ubicación"})
    df = df[df['city'] != "Otras ciudades"]

    ordered_cities = df['city'].value_counts().index.tolist()
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

df, ordered_cities = load_data()
model, scaler, metadata = load_model()
role_col = 'role_category' if 'role_category' in df.columns else 'role'

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
st.sidebar.metric("Ofertas", f"{len(df):,}")
st.sidebar.metric("Ciudades", df['city'].nunique())
st.sidebar.metric("Empresas", df['company'].nunique())

# =========================
# 🏠 OVERVIEW
# =========================
if page == "🏠 Overview":
    st.title("📊 Mercado Laboral de Data Science en España")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Ofertas", f"{len(df):,}", f"{df['is_ai_related'].sum()} IA/ML")
    with col2:
        st.metric("Salario Promedio", f"{df['salary_avg'].mean():,.0f}€")
    with col3:
        st.metric("Skills Promedio", f"{df['num_skills'].mean():.1f}")
    with col4:
        ai_percentage = (df['is_ai_related'].sum() / len(df)) * 100
        st.metric("Ofertas IA/ML", f"{ai_percentage:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Distribución de Roles")
        role_col = 'role_category' if 'role_category' in df.columns else 'role'
        role_counts = df[role_col].value_counts()
        fig = px.bar(
            x=role_counts.values, y=role_counts.index, orientation='h',
            labels={'x': 'Número de Ofertas', 'y': ''},
            color=role_counts.values, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🏙️ Top 10 Ciudades")
        city_counts = df['city'].value_counts().head(10)
        fig = px.bar(x=city_counts.index, y=city_counts.values,
                     color=city_counts.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    if 'created' in df.columns:
        st.subheader("📈 Evolución Temporal de Ofertas")
        df_temporal = df.set_index('created').resample('M').size()
        fig = px.line(x=df_temporal.index, y=df_temporal.values,
                      labels={'x': 'Fecha', 'y': 'Número de Ofertas'},
                      markers=True)
        fig.update_traces(line_color='#3498db', line_width=3)
        st.plotly_chart(fig, use_container_width=True)

# =========================
# 💼 ANÁLISIS DE ROLES
# =========================
elif page == "💼 Análisis de Roles":
    st.title("💼 Análisis de Roles y Experiencia")
    role_col = 'role_category' if 'role_category' in df.columns else 'role'

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Distribución por Nivel")
        if 'seniority' in df.columns:
            fig = px.pie(df, names='seniority', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("💼 Roles por Nivel")
        if 'seniority' in df.columns:
            role_sen = pd.crosstab(df[role_col], df['seniority'])
            fig = px.bar(role_sen, barmode='stack')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Tabla Detallada por Rol")
    if 'salary_avg' in df.columns:
        stats = df.groupby(role_col).agg({'id':'count','salary_avg':'mean','num_skills':'mean','is_ai_related':'sum'}).round(0)
        stats.columns = ['Ofertas','Salario Medio (€)','Skills Promedio','Ofertas IA/ML']
        st.dataframe(stats.sort_values('Ofertas', ascending=False), use_container_width=True)

# =========================
# 🗺️ ANÁLISIS GEOGRÁFICO
# =========================
elif page == "🗺️ Análisis Geográfico":
    st.title("🗺️ Análisis Geográfico")
    city = st.selectbox("Selecciona una ciudad:", ["Todas"] + ordered_cities)
    df_city = df if city == "Todas" else df[df['city'] == city]

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Ofertas", len(df_city))
    with col2: st.metric("Empresas", df_city['company'].nunique())
    with col3: st.metric("Salario Medio", f"{df_city['salary_avg'].mean():,.0f}€")
    with col4:
        ai_pct = (df_city['is_ai_related'].sum() / len(df_city)*100) if len(df_city)>0 else 0
        st.metric("% IA/ML", f"{ai_pct:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💼 Roles")
        fig = px.pie(df_city, names=role_col)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🏢 Top Empresas")
        top_comp = df_city['company'].value_counts().head(10)
        fig = px.bar(x=top_comp.values, y=top_comp.index, orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔥 SKILLS DEMANDADAS
# =========================
elif page == "🔥 Skills Demandadas":
    st.title("🔥 Skills Más Demandadas")
    all_skills = [s for lst in df['skills'] for s in lst]
    skill_counts = Counter(all_skills)
    top_n = st.slider("Top N:", 5, 30, 15)
    top_sk = dict(skill_counts.most_common(top_n))
    fig = px.bar(x=list(top_sk.values()), y=list(top_sk.keys()), orientation='h',
                 color=list(top_sk.values()), color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 💰 ANÁLISIS SALARIAL
# =========================
elif page == "💰 Análisis Salarial":
    st.title("💰 Análisis de Salarios")
    df_sal = df[df['salary_avg'].notna()]
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Promedio", f"{df_sal['salary_avg'].mean():,.0f}€")
    with col2: st.metric("Mediana", f"{df_sal['salary_avg'].median():,.0f}€")
    with col3: st.metric("Mínimo", f"{df_sal['salary_avg'].min():,.0f}€")
    with col4: st.metric("Máximo", f"{df_sal['salary_avg'].max():,.0f}€")

    fig = px.histogram(df_sal, x='salary_avg', nbins=25)
    fig.add_vline(x=df_sal['salary_avg'].mean(), line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🤖 IA/ML TRENDS
# =========================
elif page == "🤖 IA/ML Trends":
    st.title("🤖 Tendencias en IA y ML")
    ai_jobs = df[df['is_ai_related']==True]
    other_jobs = df[df['is_ai_related']==False]
    st.metric("Ofertas IA/ML", len(ai_jobs))
    st.metric("% del total", f"{len(ai_jobs)/len(df)*100:.1f}%")
    fig = px.bar(x=['IA/ML','Otros'], y=[ai_jobs['salary_avg'].mean(), other_jobs['salary_avg'].mean()],
                 labels={'x':'Categoría','y':'Salario medio (€)'})
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔮 PREDICTOR DE SALARIOS
# =========================
elif page == "🔮 Predictor de Salarios":
    st.title("🔮 Predictor de Salarios")
    if model is None or metadata is None:
        st.error("❌ Modelo no disponible. Sube los archivos a `/models/`.")
    else:
        st.success("✅ Modelo cargado correctamente")
        st.metric("MAE", f"{metadata['metrics']['mae']:,.0f}€")
        st.metric("R²", f"{metadata['metrics']['r2']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        skills = sorted({s.replace('skill_','') for s in metadata.get('skill_columns',[])})
        selected_skills = st.multiselect("Skills:", skills, default=['Python','SQL'])
        city = st.selectbox("Ciudad:", ['Madrid','Barcelona','Valencia','Bilbao','Sevilla','Otras'])
    with col2:
        seniority = st.selectbox("Nivel:", ['Junior','Mid-Level','Senior','Manager'])
        role = st.selectbox("Rol:", df[role_col].unique())
        is_ai = st.checkbox("¿IA/ML?", True)

    if st.button("Predecir"):
        base = df['salary_avg'].mean()
        mult = 1
        if seniority=='Senior': mult=1.2
        elif seniority=='Manager': mult=1.4
        elif seniority=='Junior': mult=0.8
        if city in ['Madrid','Barcelona']: mult*=1.1
        if is_ai: mult*=1.05
        salary = base*mult + len(selected_skills)*1000
        st.success(f"💶 Salario estimado: {salary:,.0f}€/año")

# =========================
# 📜 FOOTER
# =========================
st.markdown("---")
st.markdown("<div style='text-align:center;color:#888;'>📊 Dashboard del Mercado Laboral de Data Science en España · Datos via Adzuna API</div>", unsafe_allow_html=True)
