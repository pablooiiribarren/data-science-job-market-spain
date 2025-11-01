"""
Modelo predictivo de salarios para roles de Data Science
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Detectar carpeta raíz
if Path("data").exists():
    BASE_DIR = Path(".")
elif Path("../data").exists():
    BASE_DIR = Path("..")
else:
    BASE_DIR = Path(".")

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

class SalaryPredictor:
    """
    Modelo para predecir salarios en roles de Data Science
    """
    
    def __init__(self, data_path=None):
        """Inicializa el predictor"""
        if data_path is None:
            data_path = PROCESSED_DIR / "jobs_cleaned.csv"
        
        self.df = pd.read_csv(data_path)
        self.df['skills'] = self.df['skills'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else []
        )
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.skill_columns = []
        
        print(f"📊 Dataset cargado: {len(self.df)} ofertas")
    
    def prepare_features(self):
        """
        Prepara features para el modelo
        """
        print("\n🔧 Preparando features...")
        
        # Filtrar solo ofertas con salario
        df_model = self.df[self.df['salary_avg'].notna()].copy()
        print(f"   • Ofertas con salario: {len(df_model)}")
        
        if len(df_model) < 50:
            raise ValueError("❌ Insuficientes datos con salario para entrenar modelo (mínimo 50)")
        
        # 1. Skills como features binarias (Top 20)
        all_skills = [skill for skills_list in df_model['skills'] if skills_list for skill in skills_list]
        from collections import Counter
        top_skills = [skill for skill, _ in Counter(all_skills).most_common(20)]
        
        for skill in top_skills:
            df_model[f'skill_{skill}'] = df_model['skills'].apply(lambda x: 1 if skill in x else 0)
        
        self.skill_columns = [f'skill_{skill}' for skill in top_skills]
        print(f"   • Top {len(top_skills)} skills como features")
        
        # 2. Número total de skills
        df_model['total_skills'] = df_model['skills'].apply(len)
        
        # 3. Nivel de experiencia (ordinal)
        seniority_map = {
            'Junior': 1,
            'Mid-Level': 2,
            'Senior': 3,
            'Manager': 4,
            'No especificado': 2  # Default a mid-level
        }
        df_model['seniority_encoded'] = df_model['seniority'].map(seniority_map).fillna(2).astype(int)
        
        # 4. Categoría de rol (one-hot encoding)
        role_dummies = pd.get_dummies(df_model['role_category'], prefix='role', dtype=int)
        role_columns = role_dummies.columns.tolist()
        df_model = pd.concat([df_model, role_dummies], axis=1)
        
        # 5. Ciudad (one-hot encoding, solo top 10)
        top_cities = df_model['city'].value_counts().head(10).index
        df_model['city_group'] = df_model['city'].apply(lambda x: x if x in top_cities else 'Otras')
        city_dummies = pd.get_dummies(df_model['city_group'], prefix='city', dtype=int)
        city_columns = city_dummies.columns.tolist()
        df_model = pd.concat([df_model, city_dummies], axis=1)
        
        # 6. IA/ML flag
        df_model['is_ai'] = df_model['is_ai_related'].astype(int)
        
        # Seleccionar SOLO las columnas que sabemos que son numéricas
        feature_cols = (
            self.skill_columns + 
            ['total_skills', 'seniority_encoded', 'is_ai'] +
            role_columns +
            city_columns
        )
        
        # Extraer X y y
        X = df_model[feature_cols].copy()
        y = df_model['salary_avg'].copy()
        
        # Verificar tipos de datos
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"   ⚠️  Columnas no numéricas detectadas: {non_numeric}")
            for col in non_numeric:
                X[col] = X[col].astype(int)
        
        # Verificar valores nulos
        if X.isnull().any().any():
            print("   ⚠️  Rellenando valores nulos con 0")
            X = X.fillna(0)
        
        # Asegurar que todo es int o float
        X = X.astype(float)
        
        self.feature_names = feature_cols
        
        print(f"   • Total features: {len(feature_cols)}")
        print(f"   • Samples: {len(X)}")
        print(f"   • Verificación: Todas las columnas numéricas ✓")
        
        return X, y, df_model
    
    def train_models(self, X, y):
        """
        Entrena y compara múltiples modelos
        """
        print("\n🤖 Entrenando modelos...")
        print("=" * 60)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modelos a probar
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'Ridge Regression': Ridge(alpha=10)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n📊 {name}:")
            
            # Entrenar
            if name == 'Ridge Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"   • MAE: {mae:,.0f}€")
            print(f"   • RMSE: {rmse:,.0f}€")
            print(f"   • R²: {r2:.3f}")
        
        # Seleccionar mejor modelo (por R²)
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        self.model = results[best_model_name]['model']
        
        print("\n" + "=" * 60)
        print(f"✅ Mejor modelo: {best_model_name}")
        print(f"   R² = {results[best_model_name]['r2']:.3f}")
        
        return results, X_test, y_test, best_model_name
    
    def feature_importance(self, model_name):
        """Analiza importancia de features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            print("\n📊 Top 15 Features Más Importantes:")
            print("-" * 60)
            for i, idx in enumerate(indices, 1):
                feature = self.feature_names[idx]
                importance = importances[idx]
                print(f"{i:2}. {feature:30} {importance:.4f}")
            
            # Visualización
            plt.figure(figsize=(12, 8))
            plt.barh(range(15), importances[indices][::-1], color=sns.color_palette("viridis", 15))
            plt.yticks(range(15), [self.feature_names[i] for i in indices][::-1])
            plt.xlabel('Importancia', fontsize=12, fontweight='bold')
            plt.title('Top 15 Features Más Importantes para Predicción de Salario', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(BASE_DIR / 'images' / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\n✅ Gráfico guardado: images/feature_importance.png")
    
    def plot_predictions(self, y_test, y_pred, model_name):
        """Visualiza predicciones vs reales"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax1.scatter(y_test, y_pred, alpha=0.5, s=50)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Predicción perfecta')
        ax1.set_xlabel('Salario Real (€)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Salario Predicho (€)', fontsize=12, fontweight='bold')
        ax1.set_title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Residuos
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=50)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Salario Predicho (€)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuos (€)', fontsize=12, fontweight='bold')
        ax2.set_title('Análisis de Residuos', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'Evaluación del Modelo: {model_name}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(BASE_DIR / 'images' / 'model_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico guardado: images/model_predictions.png")
    
    def save_model(self, model_name, metrics):
        """Guarda el modelo y metadatos"""
        # Guardar modelo
        model_path = MODELS_DIR / 'salary_predictor.pkl'
        joblib.dump(self.model, model_path)
        
        # Guardar scaler
        scaler_path = MODELS_DIR / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Guardar metadata
        metadata = {
            'model_type': model_name,
            'features': self.feature_names,
            'metrics': {
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'r2': float(metrics['r2'])
            },
            'skill_columns': self.skill_columns,
            'trained_on': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = MODELS_DIR / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Modelo guardado en: {MODELS_DIR}")
        print(f"   • {model_path.name}")
        print(f"   • {scaler_path.name}")
        print(f"   • {metadata_path.name}")
    
    def predict_salary(self, skills, city, seniority, role_category, is_ai=False):
        """
        Predice salario para un perfil dado
        
        Args:
            skills: Lista de skills (ej: ['Python', 'SQL', 'AWS'])
            city: Ciudad (ej: 'Madrid')
            seniority: Nivel (ej: 'Senior')
            role_category: Tipo de rol (ej: 'Data Scientist')
            is_ai: Si es rol de IA/ML
        
        Returns:
            float: Salario predicho
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train() primero.")
        
        # Crear feature vector
        features = {}
        
        # Skills
        for skill_col in self.skill_columns:
            skill_name = skill_col.replace('skill_', '')
            features[skill_col] = 1 if skill_name in skills else 0
        
        # Total skills
        features['total_skills'] = len(skills)
        
        # Seniority
        seniority_map = {'Junior': 1, 'Mid-Level': 2, 'Senior': 3, 'Manager': 4}
        features['seniority_encoded'] = seniority_map.get(seniority, 2)
        
        # IA flag
        features['is_ai'] = 1 if is_ai else 0
        
        # Roles (todas a 0 excepto la seleccionada)
        for feature in self.feature_names:
            if feature.startswith('role_'):
                features[feature] = 1 if feature == f'role_{role_category}' else 0
        
        # Ciudades (todas a 0 excepto la seleccionada)
        for feature in self.feature_names:
            if feature.startswith('city_'):
                city_name = feature.replace('city_', '')
                features[feature] = 1 if city_name == city or (city not in ['Madrid', 'Barcelona', 'Valencia'] and city_name == 'Otras') else 0
        
        # Convertir a DataFrame
        X_pred = pd.DataFrame([features])[self.feature_names]
        
        # Predecir
        prediction = self.model.predict(X_pred)[0]
        
        return prediction
    
    def train(self):
        """Pipeline completo de entrenamiento"""
        print("\n" + "🎯 ENTRENAMIENTO DEL MODELO PREDICTIVO ".center(60, "="))
        
        # Preparar datos
        X, y, df_model = self.prepare_features()
        
        # Entrenar modelos
        results, X_test, y_test, best_model_name = self.train_models(X, y)
        
        # Análisis
        self.feature_importance(best_model_name)
        
        # Visualizaciones
        y_pred = results[best_model_name]['predictions']
        self.plot_predictions(y_test, y_pred, best_model_name)
        
        # Guardar
        self.save_model(best_model_name, results[best_model_name])
        
        print("\n" + "=" * 60)
        print("✅ Entrenamiento completado exitosamente")
        print("🎯 Siguiente paso: Dashboard interactivo con Streamlit")
        print("=" * 60)
        
        return results[best_model_name]


def main():
    """Función principal"""
    try:
        predictor = SalaryPredictor()
        metrics = predictor.train()
        
        # Ejemplo de predicción
        print("\n" + "=" * 60)
        print("🔮 EJEMPLO DE PREDICCIÓN")
        print("=" * 60)
        
        example_salary = predictor.predict_salary(
            skills=['Python', 'SQL', 'Machine Learning', 'AWS'],
            city='Madrid',
            seniority='Senior',
            role_category='Data Scientist',
            is_ai=True
        )
        
        print("\n📊 Perfil ejemplo:")
        print("   • Skills: Python, SQL, Machine Learning, AWS")
        print("   • Ciudad: Madrid")
        print("   • Nivel: Senior")
        print("   • Rol: Data Scientist (IA/ML)")
        print(f"\n💰 Salario predicho: {example_salary:,.0f}€/año")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()