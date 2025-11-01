"""
Limpieza de ubicaciones para visualizaciones del dashboard
"""

import pandas as pd
from pathlib import Path

# Rutas
DATA_PATH = Path("data/processed/jobs_cleaned.csv")

# Cargar datos
df = pd.read_csv(DATA_PATH)

print(f"📂 Cargando {len(df)} ofertas...")

# Normalizar nombres
df['city'] = df['city'].str.strip()

# Reasignar municipios a ciudades principales
mapping = {
    "Alcobendas": "Madrid",
    "Boadilla del Monte": "Madrid",
    "Sant Adrià de Besòs": "Barcelona",
    "Sant Cugat del Vallès": "Barcelona",
    "Pozuelo de Alarcón": "Madrid",
    "Paracuellos de Jarama": "Madrid", 
    "Esplugues de Llobregat": "Barcelona"
}

df['city'] = df['city'].replace(mapping)

# Cambiar nombre de grupo remoto
df['city'] = df['city'].replace({
    "Remoto/Sin especificar": "Remoto o sin ubicación"
})

# Eliminar categoría genérica
if "Otras ciudades" in df['city'].values:
    print("🧹 Eliminando categoría genérica 'Otras ciudades'...")
    df = df[df['city'] != "Otras ciudades"]

# Mostrar resumen final
print("\n✅ Ciudades finales:")
print(df['city'].value_counts().head(15))
print(f"\nTotal ciudades únicas: {df['city'].nunique()}")

# Guardar resultado limpio
clean_path = Path("data/processed/jobs_cleaned_cleaned.csv")
df.to_csv(clean_path, index=False)
print(f"\n💾 CSV limpio guardado en: {clean_path}")
