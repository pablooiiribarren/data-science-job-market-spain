"""
Script para verificar que todo está correctamente instalado
"""

def test_imports():
    """Prueba que todas las librerías se importan correctamente"""
    print("🧪 Probando imports...")
    
    try:
        import pandas as pd
        print("✅ pandas")
        
        import numpy as np
        print("✅ numpy")
        
        import requests
        print("✅ requests")
        
        import plotly.express as px
        print("✅ plotly")
        
        import streamlit as st
        print("✅ streamlit")
        
        import sklearn
        print("✅ scikit-learn")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv")
        
        print("\n🎉 ¡Todas las librerías están instaladas correctamente!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("Ejecuta: pip install -r requirements.txt")
        return False

def test_config():
    """Prueba que la configuración se carga correctamente"""
    print("\n🔧 Probando configuración...")
    
    try:
        from src.config import CITIES, SKILLS_LIST, ADZUNA_APP_ID
        print(f"✅ Ciudades configuradas: {list(CITIES.keys())}")
        print(f"✅ Skills a buscar: {len(SKILLS_LIST)} skills")
        
        if ADZUNA_APP_ID:
            print("✅ API Key de Adzuna detectada")
        else:
            print("⚠️  API Key de Adzuna no configurada (necesaria para Fase 2)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_directories():
    """Verifica que los directorios existen"""
    print("\n📁 Verificando estructura de directorios...")
    
    from pathlib import Path
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/sample",
        "notebooks",
        "src",
        "app"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - no existe")
            all_exist = False
            
    return all_exist

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 VERIFICACIÓN DEL ENTORNO")
    print("=" * 50)
    
    test_imports()
    test_config()
    test_directories()
    
    print("\n" + "=" * 50)
    print("✨ Verificación completada")
    print("=" * 50)