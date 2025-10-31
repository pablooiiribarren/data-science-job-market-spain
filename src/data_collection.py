"""
Script principal para recolectar ofertas de trabajo de Data Science
Fuente: Adzuna API
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from config import (
    ADZUNA_APP_ID, 
    ADZUNA_API_KEY, 
    SEARCH_TERMS,
    RAW_DATA_DIR
)

class AdzunaJobCollector:
    """
    Clase para recolectar ofertas de trabajo desde Adzuna API
    """
    
    def __init__(self):
        self.app_id = ADZUNA_APP_ID
        self.app_key = ADZUNA_API_KEY
        self.base_url = "https://api.adzuna.com/v1/api/jobs/es/search"
        self.jobs_collected = []
        
    def search_jobs(self, search_term, location=None, page=1, results_per_page=50):
        """
        Realiza una búsqueda de ofertas de trabajo
        
        Args:
            search_term: Término de búsqueda (ej: "Data Scientist")
            location: Ubicación (opcional, None = toda España)
            page: Número de página
            results_per_page: Resultados por página (máx 50)
        
        Returns:
            dict: Respuesta JSON de la API
        """
        url = f"{self.base_url}/{page}"
        
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": search_term,
            "results_per_page": results_per_page,
            "content-type": "application/json"
        }
        
        if location:
            params["where"] = location
        
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("⚠️  Rate limit alcanzado. Esperando 60 segundos...")
                time.sleep(60)
                return self.search_jobs(search_term, location, page, results_per_page)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return None
    
    def parse_job(self, job_data):
        """
        Extrae y estructura la información de una oferta
        
        Args:
            job_data: Datos crudos de la oferta desde la API
        
        Returns:
            dict: Oferta estructurada
        """
        return {
            # IDs y timestamps
            "id": job_data.get("id"),
            "created": job_data.get("created"),
            "collected_at": datetime.now().isoformat(),
            
            # Información básica
            "title": job_data.get("title"),
            "description": job_data.get("description"),
            "category": job_data.get("category", {}).get("label"),
            "contract_type": job_data.get("contract_type"),
            "contract_time": job_data.get("contract_time"),
            
            # Empresa
            "company_name": job_data.get("company", {}).get("display_name"),
            
            # Ubicación
            "location_display": job_data.get("location", {}).get("display_name"),
            "location_area": job_data.get("location", {}).get("area", []),
            "latitude": job_data.get("latitude"),
            "longitude": job_data.get("longitude"),
            
            # Salario
            "salary_min": job_data.get("salary_min"),
            "salary_max": job_data.get("salary_max"),
            "salary_is_predicted": job_data.get("salary_is_predicted"),
            
            # URLs
            "redirect_url": job_data.get("redirect_url"),
            
            # Metadata
            "adref": job_data.get("adref"),
        }
    
    def collect_all_jobs(self, search_terms=None, locations=None, max_pages=20):
        """
        Recolecta ofertas de trabajo de forma exhaustiva
        
        Args:
            search_terms: Lista de términos a buscar
            locations: Lista de ubicaciones (None = toda España)
            max_pages: Máximo de páginas por búsqueda
        """
        if search_terms is None:
            search_terms = SEARCH_TERMS
        
        if locations is None:
            locations = [None]  # Buscar en toda España
        
        print("🚀 Iniciando recolección de datos...")
        print(f"📋 Términos de búsqueda: {len(search_terms)}")
        print(f"📍 Ubicaciones: {len(locations)}")
        print("=" * 70)
        
        total_searches = len(search_terms) * len(locations)
        search_count = 0
        
        for search_term in search_terms:
            for location in locations:
                search_count += 1
                location_name = location if location else "Toda España"
                
                print(f"\n[{search_count}/{total_searches}] 🔍 '{search_term}' en {location_name}")
                
                # Primera búsqueda para saber el total
                first_page = self.search_jobs(search_term, location, page=1)
                
                if not first_page:
                    print("   ⚠️  Sin resultados o error")
                    continue
                
                total_results = first_page.get("count", 0)
                print(f"   📊 Total disponibles: {total_results}")
                
                if total_results == 0:
                    continue
                
                # Calcular cuántas páginas necesitamos
                pages_needed = min(
                    (total_results // 50) + 1,
                    max_pages
                )
                
                # Recolectar todas las páginas
                for page in tqdm(range(1, pages_needed + 1), 
                               desc=f"   Páginas", 
                               leave=False):
                    
                    if page == 1:
                        data = first_page
                    else:
                        data = self.search_jobs(search_term, location, page)
                        time.sleep(1)  # Respetar rate limits
                    
                    if not data:
                        continue
                    
                    # Procesar resultados
                    for job in data.get("results", []):
                        parsed_job = self.parse_job(job)
                        parsed_job["search_term"] = search_term
                        parsed_job["search_location"] = location_name
                        self.jobs_collected.append(parsed_job)
                
                print(f"   ✅ {len(self.jobs_collected)} ofertas acumuladas")
        
        print("\n" + "=" * 70)
        print(f"✨ Recolección completada: {len(self.jobs_collected)} ofertas totales")
        
        return self.jobs_collected
    
    def save_data(self, filename=None):
        """
        Guarda los datos recolectados en CSV y JSON
        
        Args:
            filename: Nombre del archivo (sin extensión)
        """
        if not self.jobs_collected:
            print("⚠️  No hay datos para guardar")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jobs_data_{timestamp}"
        
        # Convertir a DataFrame
        df = pd.DataFrame(self.jobs_collected)
        
        # Eliminar duplicados por ID
        df_unique = df.drop_duplicates(subset=['id'], keep='first')
        duplicates_removed = len(df) - len(df_unique)
        
        if duplicates_removed > 0:
            print(f"🗑️  {duplicates_removed} duplicados eliminados")
        
        # Guardar CSV
        csv_path = RAW_DATA_DIR / f"{filename}.csv"
        df_unique.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"💾 CSV guardado: {csv_path}")
        
        # Guardar JSON (más detallado)
        json_path = RAW_DATA_DIR / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.jobs_collected, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON guardado: {json_path}")
        
        # Estadísticas básicas
        print("\n📊 Resumen de datos:")
        print(f"   Total ofertas únicas: {len(df_unique)}")
        print(f"   Ofertas con salario: {df_unique['salary_min'].notna().sum()}")
        print(f"   Ubicaciones únicas: {df_unique['location_display'].nunique()}")
        print(f"   Empresas únicas: {df_unique['company_name'].nunique()}")
        
        return df_unique


def main():
    """
    Función principal para ejecutar la recolección
    """
    print("\n" + "🎯 DATA SCIENCE JOB COLLECTOR ".center(70, "="))
    print()
    
    # Crear el collector
    collector = AdzunaJobCollector()
    
    # Estrategia de búsqueda ampliada
    search_terms = [
        "Data Scientist",
        "Data Analyst", 
        "Machine Learning Engineer",
        "Data Engineer",
        "AI Engineer",
        "Business Intelligence Analyst",
        "Analytics Engineer",
        "MLOps Engineer"
    ]
    
    # Buscar en toda España (más ciudades importantes)
    locations = [
        None,  # Toda España
        "Madrid",
        "Barcelona", 
        "Valencia",
        "Sevilla",
        "Bilbao",
        "Málaga"
    ]
    
    # Recolectar datos
    jobs = collector.collect_all_jobs(
        search_terms=search_terms,
        locations=locations,
        max_pages=10  # Máximo 10 páginas por búsqueda
    )
    
    # Guardar datos
    if jobs:
        df = collector.save_data()
        
        print("\n" + "=" * 70)
        print("✅ Proceso completado exitosamente")
        print("🎯 Siguiente paso: Limpieza y análisis exploratorio")
        print("=" * 70)
        
        return df
    else:
        print("\n⚠️  No se recolectaron datos. Revisa tu conexión y credenciales.")
        return None


if __name__ == "__main__":
    df = main()