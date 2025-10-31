"""
Script para probar la conexión con Adzuna API
"""
import requests
from config import ADZUNA_APP_ID, ADZUNA_API_KEY, CITIES

def test_adzuna_connection():
    """
    Prueba la conexión con Adzuna API
    """
    print("🔌 Probando conexión con Adzuna API...")
    print("=" * 60)
    
    # Verificar credenciales
    if not ADZUNA_APP_ID or not ADZUNA_API_KEY:
        print("❌ Error: Credenciales no configuradas en .env")
        print("Por favor, añade ADZUNA_APP_ID y ADZUNA_API_KEY")
        return False
    
    print(f"✅ App ID: {ADZUNA_APP_ID[:8]}...")
    print(f"✅ API Key: {ADZUNA_API_KEY[:8]}...")
    
    # URL base de Adzuna para España
    base_url = "https://api.adzuna.com/v1/api/jobs/es/search/1"
    
    # Parámetros de búsqueda
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_API_KEY,
        "what": "data scientist",  # Búsqueda simple de prueba
        "where": "Bilbao",
        "results_per_page": 5,
        "content-type": "application/json"
    }
    
    print(f"\n🔍 Buscando: '{params['what']}' en '{params['where']}'")
    print("-" * 60)
    
    try:
        # Hacer la petición
        response = requests.get(base_url, params=params, timeout=10)
        
        # Verificar status code
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Conexión exitosa!")
            print(f"📊 Total de resultados disponibles: {data.get('count', 0)}")
            print(f"📄 Resultados en esta página: {len(data.get('results', []))}")
            
            # Mostrar algunos resultados
            print("\n📋 Primeras ofertas encontradas:")
            print("-" * 60)
            
            for i, job in enumerate(data.get('results', [])[:3], 1):
                print(f"\n{i}. {job.get('title', 'Sin título')}")
                print(f"   🏢 Empresa: {job.get('company', {}).get('display_name', 'No especificada')}")
                print(f"   📍 Ubicación: {job.get('location', {}).get('display_name', 'No especificada')}")
                
                # Salario (si está disponible)
                salary_min = job.get('salary_min')
                salary_max = job.get('salary_max')
                if salary_min and salary_max:
                    print(f"   💰 Salario: {salary_min:,.0f}€ - {salary_max:,.0f}€")
                else:
                    print(f"   💰 Salario: No especificado")
                
                print(f"   🔗 URL: {job.get('redirect_url', 'N/A')[:60]}...")
            
            print("\n" + "=" * 60)
            print("✨ ¡API funcionando correctamente! Listo para recolectar datos.")
            return True
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Error: Tiempo de espera agotado")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en la petición: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_all_cities():
    """
    Prueba la API con todas las ciudades configuradas
    """
    print("\n\n🌍 Probando búsquedas en todas las ciudades...")
    print("=" * 60)
    
    base_url = "https://api.adzuna.com/v1/api/jobs/es/search/1"
    
    for city_key, city_info in CITIES.items():
        print(f"\n📍 {city_info['location']}:")
        
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_API_KEY,
            "what": "data analyst",
            "where": city_info['location'],
            "results_per_page": 1,
            "content-type": "application/json"
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                print(f"   ✅ {count} ofertas encontradas")
            else:
                print(f"   ⚠️  Error {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("\n" + "🚀 TEST DE ADZUNA API " + "\n")
    
    # Test básico
    success = test_adzuna_connection()
    
    if success:
        # Test con todas las ciudades
        test_all_cities()
        
        print("\n" + "=" * 60)
        print("✅ Todos los tests completados")
        print("🎯 Siguiente paso: Crear el script de recolección masiva")
        print("=" * 60)
    else:
        print("\n⚠️  Por favor, revisa tus credenciales en .env")