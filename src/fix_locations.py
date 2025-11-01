"""
Script para limpiar y corregir las ubicaciones en el dataset
Elimina 'España' como ciudad y categoriza ubicaciones genéricas
"""

import pandas as pd
from pathlib import Path

# Rutas
BASE_DIR = Path(".")
if not (BASE_DIR / "data").exists():
    BASE_DIR = Path("..")

PROCESSED_DIR = BASE_DIR / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "jobs_cleaned.csv"
OUTPUT_FILE = PROCESSED_DIR / "jobs_cleaned.csv"

def fix_locations():
    """Corrige las ubicaciones en el dataset"""
    
    print("🔧 Corrigiendo ubicaciones...")
    print("=" * 60)
    
    # Cargar datos
    df = pd.read_csv(INPUT_FILE)
    print(f"📊 Total ofertas: {len(df)}")
    
    # Ver distribución actual de ciudades
    print("\n📍 Top 15 ubicaciones antes de limpiar:")
    print(df['city'].value_counts().head(15))
    
    # Identificar ubicaciones genéricas a limpiar
    generic_locations = ['España', 'Spain', 'Espana', 'Remote', 'Remoto', 
                        'Teletrabajo', 'Desconocido', 'No especificada']
    
    # Contar cuántas hay
    generic_count = df['city'].isin(generic_locations).sum()
    print(f"\n⚠️  Ubicaciones genéricas encontradas: {generic_count}")
    
    # Opción 1: Marcarlas como "Remoto/Sin especificar"
    df.loc[df['city'].isin(generic_locations), 'city'] = 'Remoto/Sin especificar'
    
    # Opción 2: También limpiar ubicaciones que sean demasiado genéricas
    # (menos de 5 ofertas se agrupan como "Otras ciudades")
    city_counts = df['city'].value_counts()
    small_cities = city_counts[city_counts < 5].index.tolist()
    
    # No agrupar "Remoto/Sin especificar" con otras
    if 'Remoto/Sin especificar' in small_cities:
        small_cities.remove('Remoto/Sin especificar')
    
    print(f"\n📌 Ciudades con <5 ofertas: {len(small_cities)}")
    df.loc[df['city'].isin(small_cities), 'city'] = 'Otras ciudades'
    
    # Guardar
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print("\n✅ Datos actualizados guardados")
    print("\n📍 Top 15 ubicaciones después de limpiar:")
    print(df['city'].value_counts().head(15))
    
    print("\n" + "=" * 60)
    print("✅ Corrección completada")
    print(f"💾 Archivo guardado: {OUTPUT_FILE}")
    print("\n🔄 Reinicia el dashboard de Streamlit para ver los cambios")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    df = fix_locations()