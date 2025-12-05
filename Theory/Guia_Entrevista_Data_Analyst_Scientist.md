# 📊 GUÍA COMPLETA DE ESTUDIO: ENTREVISTA TÉCNICA PARA DATA ANALYST Y DATA SCIENTIST

## Versión 1.0 | Especialista en IT Hiring

---

## 🎯 INTRODUCCIÓN Y PROPÓSITO

Esta guía está diseñada para preparar profesionales que aspiren a roles de **Data Analyst Junior/Senior** y **Data Scientist**. Cubriremos el **flujo completo del pipeline de datos**, desde la carga inicial de datos hasta las recomendaciones finales basadas en análisis y modelado de machine learning.

El enfoque es **práctico y orientado a entrevistas reales**: incluye explicaciones de conceptos, comandos de Python con ejemplos típicos, y consideraciones que los hiring managers evalúan en candidatos.

---

## 📋 CONTENIDO PRINCIPAL

### 1️⃣ FASE 1: CARGA Y EXPLORACIÓN DE DATOS

#### 1.1 Concepto General
La carga de datos es el primer paso en cualquier pipeline analítico. Requiere:
- Identificar la fuente de datos (archivos locales, bases de datos, APIs, streaming)
- Conectarse a esa fuente
- Cargar los datos en una estructura manejable

#### 1.2 Comandos Python Típicos

**Importar librerías necesarias:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

**Cargar datos desde CSV:**
```python
# Método básico
df = pd.read_csv('datos.csv')

# Con parámetros útiles
df = pd.read_csv('datos.csv', 
                  sep=',',                    # Delimitador
                  encoding='utf-8',           # Codificación
                  nrows=1000,                 # Primeras N filas
                  skiprows=5)                 # Saltar primeras 5 filas
```

**Cargar desde otras fuentes:**
```python
# Desde Excel
df = pd.read_excel('datos.xlsx', sheet_name='Sheet1')

# Desde base de datos (ej: MySQL con SQLAlchemy)
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://usuario:contraseña@localhost/bd_nombre')
df = pd.read_sql('SELECT * FROM tabla_nombre', con=engine)

# Desde JSON
df = pd.read_json('datos.json')
```

#### 1.3 Exploración Inicial (Exploratory Data Analysis - EDA)

**Primeras observaciones:**
```python
# Ver primeras filas
df.head()          # Primeras 5 filas (default)
df.head(10)        # Primeras 10 filas

# Ver últimas filas
df.tail()

# Información general del dataset
df.info()          # Tipos de datos, valores no-nulos, memoria
df.describe()      # Estadísticas descriptivas (media, std, cuartiles)
df.shape           # Dimensiones (filas, columnas)

# Estadísticas personalizadas
df.describe(percentiles=[.25, .5, .75, .99])  # Percentiles específicos
```

**Análisis de datos faltantes:**
```python
# Detectar valores faltantes
df.isnull().sum()                    # Conteo de NaN por columna
df.isnull().sum() / len(df) * 100    # Porcentaje de valores faltantes
df.dropna()                          # Dataset sin valores faltantes

# Visualizar patrón de faltantes
import seaborn as sns
sns.heatmap(df.isnull())
```

**Análisis de datos duplicados:**
```python
# Detectar duplicados
df.duplicated().sum()                # Número de filas duplicadas
df.drop_duplicates()                 # Eliminar duplicados
df.drop_duplicates(subset=['col1'])  # Considerar solo ciertas columnas
```

**Análisis de tipos de datos:**
```python
df.dtypes                            # Tipos de datos por columna
df['columna'].value_counts()         # Frecuencia de valores únicos
df['columna'].unique()               # Valores únicos
df.nunique()                         # Cantidad de valores únicos por columna
```

**Correlación y relaciones:**
```python
# Matriz de correlación (solo variables numéricas)
df.corr()

# Visualizar correlación
sns.heatmap(df.corr(), annot=True)
plt.show()
```

---

### 2️⃣ FASE 2: LIMPIEZA Y PREPARACIÓN DE DATOS

#### 2.1 Concepto General
La limpieza de datos es crítica (representa ~80% del trabajo real). Incluye:
- Manejo de valores faltantes
- Detección y tratamiento de outliers
- Corrección de tipos de datos
- Tratamiento de valores inconsistentes

#### 2.2 Manejo de Valores Faltantes

**Estrategia 1: Eliminación**
```python
# Eliminar filas con cualquier valor faltante
df_clean = df.dropna()

# Eliminar filas donde una columna específica es nula
df_clean = df.dropna(subset=['columna_importante'])

# Eliminar si más del 50% de valores están faltantes
df_clean = df.dropna(thresh=len(df)*0.5, axis=1)
```

**Estrategia 2: Imputación con estadísticas**
```python
# Llenar con media (variables numéricas)
df['columna_numerica'] = df['columna_numerica'].fillna(df['columna_numerica'].mean())

# Llenar con mediana (más robusta ante outliers)
df['columna_numerica'] = df['columna_numerica'].fillna(df['columna_numerica'].median())

# Llenar con moda (categorías)
df['columna_categorica'] = df['columna_categorica'].fillna(df['columna_categorica'].mode()[0])

# Llenar con valor específico
df['columna'] = df['columna'].fillna(0)
```

**Estrategia 3: Forward/Backward fill (series temporales)**
```python
# Forward fill (propagar último valor válido hacia adelante)
df['columna'] = df['columna'].fillna(method='ffill')

# Backward fill (propagar valor futuro hacia atrás)
df['columna'] = df['columna'].fillna(method='bfill')
```

#### 2.3 Detección y Tratamiento de Outliers

**Método 1: Rango Intercuartílico (IQR)**
```python
# Calcular IQR
Q1 = df['columna'].quantile(0.25)
Q3 = df['columna'].quantile(0.75)
IQR = Q3 - Q1

# Definir límites
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrar outliers
df_clean = df[(df['columna'] >= lower_bound) & (df['columna'] <= upper_bound)]

# Alternativa: reemplazar outliers con límites
df['columna'] = df['columna'].clip(lower=lower_bound, upper=upper_bound)
```

**Método 2: Z-score (desviación estándar)**
```python
from scipy import stats

# Calcular Z-score
z_scores = np.abs(stats.zscore(df['columna']))

# Filtrar con umbral (típicamente |z| > 3)
df_clean = df[z_scores < 3]
```

**Método 3: Percentiles**
```python
# Usar percentiles 1 y 99
p1 = df['columna'].quantile(0.01)
p99 = df['columna'].quantile(0.99)
df_clean = df[(df['columna'] >= p1) & (df['columna'] <= p99)]
```

#### 2.4 Corrección de Tipos de Datos

**Conversión de tipos:**
```python
# Convertir a numérico
df['columna'] = pd.to_numeric(df['columna'], errors='coerce')

# Convertir a categoría
df['columna'] = df['columna'].astype('category')

# Convertir a datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')

# Convertir a string
df['columna'] = df['columna'].astype(str)

# Convertir a booleano
df['columna'] = df['columna'].astype(bool)
```

**Limpieza de strings:**
```python
# Eliminar espacios en blanco
df['columna'] = df['columna'].str.strip()

# Convertir a minúsculas
df['columna'] = df['columna'].str.lower()

# Reemplazar caracteres
df['columna'] = df['columna'].str.replace('viejo', 'nuevo')

# Remover caracteres especiales
df['columna'] = df['columna'].str.replace(r'[^a-zA-Z0-9]', '')
```

#### 2.5 Validación de Calidad

```python
# Verificar rangos válidos
assert df['edad'].min() >= 0 and df['edad'].max() <= 120

# Verificar valores únicos esperados
assert df['genero'].isin(['M', 'F', 'Otro']).all()

# Asegurar sin valores nulos críticos
assert df['ID'].isnull().sum() == 0
```

---

### 3️⃣ FASE 3: CODIFICACIÓN Y TRANSFORMACIÓN DE VARIABLES

#### 3.1 Concepto General
Las máquinas de aprendizaje necesitan input numérico. La codificación convierte variables categóricas en representaciones numéricas apropiadas.

#### 3.2 Técnicas de Codificación

**Label Encoding (Variables Ordinales)**
```python
from sklearn.preprocessing import LabelEncoder

# Para variable con orden natural: Low < Medium < High
mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['nivel_codificado'] = df['nivel'].map(mapping)

# Alternativa con LabelEncoder (orden alfabético)
le = LabelEncoder()
df['genero_encoded'] = le.fit_transform(df['genero'])
# Puede recuperar mapping: dict(zip(le.classes_, le.transform(le.classes_)))
```

**One-Hot Encoding (Variables Nominales)**
```python
# Método 1: pd.get_dummies()
df_encoded = pd.get_dummies(df, columns=['ciudad'], prefix='ciudad')
# Resultado: ciudad_Madrid, ciudad_Barcelona, ciudad_Valencia (0 o 1)

# Método 2: sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' para evitar multicolinealidad
df_encoded = pd.DataFrame(ohe.fit_transform(df[['ciudad']]))
```

**Manejo de variables con muchas categorías:**
```python
# Agrupar categorías raras
categorias_frecuentes = df['ciudad'].value_counts().head(5).index
df['ciudad_grouped'] = df['ciudad'].apply(
    lambda x: x if x in categorias_frecuentes else 'Otras'
)
```

#### 3.3 Feature Engineering (Ingeniería de Características)

**Creación de nuevas variables:**
```python
# Variables calculadas
df['edad_al_cuadrado'] = df['edad'] ** 2
df['log_precio'] = np.log(df['precio'] + 1)  # +1 para evitar log(0)
df['ratio'] = df['columna1'] / df['columna2']

# Variables de interacción
df['edad_x_salario'] = df['edad'] * df['salario']

# Variables binarias
df['es_adulto'] = (df['edad'] >= 18).astype(int)
```

**Características temporales:**
```python
# Extraer componentes de fecha
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['dia'] = df['fecha'].dt.day
df['dia_semana'] = df['fecha'].dt.dayofweek
df['es_fin_semana'] = (df['fecha'].dt.dayofweek >= 5).astype(int)

# Características relativas
df['dias_desde_hoy'] = (pd.Timestamp.now() - df['fecha']).dt.days
```

**Agrupaciones y agregaciones:**
```python
# Estadísticas por grupo
resumen = df.groupby('categoria')['precio'].agg(['mean', 'min', 'max', 'std'])

# Merge de estadísticas al dataset original
df = df.merge(resumen, on='categoria', how='left')
```

---

### 4️⃣ FASE 4: ESCALADO Y NORMALIZACIÓN

#### 4.1 Concepto General
El escalado asegura que características con diferentes rangos no dominen el modelo. Crítico para algoritmos basados en distancia (KNN, SVM, redes neuronales).

#### 4.2 Técnicas de Escalado

**Standardization (Z-score normalization)**
```python
from sklearn.preprocessing import StandardScaler

# Transforma datos a media=0 y std=1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['edad', 'salario', 'experiencia']])
df_scaled = pd.DataFrame(df_scaled, columns=['edad', 'salario', 'experiencia'])

# Ideal para: regresión lineal, logística, redes neuronales
```

**Normalization (Min-Max Scaling)**
```python
from sklearn.preprocessing import MinMaxScaler

# Escala datos a rango [0, 1]
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['edad', 'salario']])

# Fórmula: X_scaled = (X - X_min) / (X_max - X_min)
# Ideal para: redes neuronales, KNN, distancia euclidiana
```

**Robust Scaling**
```python
from sklearn.preprocessing import RobustScaler

# Menos sensible a outliers (usa mediana e IQR)
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[['precio']])  # Si hay outliers
```

**Importante: Separar el escalado train/test**
```python
from sklearn.model_selection import train_test_split

X = df[['edad', 'salario', 'experiencia']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler SOLO en training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transform, no fit

# NUNCA hacer esto:
# scaler.fit_transform(X_test)  # ❌ Data leakage
```

---

### 5️⃣ FASE 5: DIVISIÓN TRAIN/TEST Y VALIDACIÓN CRUZADA

#### 5.1 Concepto General
Dividir datos evalúa cómo el modelo generaliza a datos no vistos.

#### 5.2 Train-Test Split

**División básica:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% para test, 80% para train
    random_state=42,         # Reproducibilidad
    stratify=y               # Mantener proporción de clases (clasificación)
)

print(f"Training: {X_train.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")
```

#### 5.3 K-Fold Cross-Validation

**Mejor para datasets pequeños:**
```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Configurar K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Opción 1: Evaluar manualmente en cada fold
model = RandomForestClassifier(random_state=42)
for train_idx, test_idx in kf.split(X):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_test_fold, y_test_fold)
    print(f"Fold accuracy: {score:.3f}")

# Opción 2: cross_val_score (automático)
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

**Variantes de validación cruzada:**
```python
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, TimeSeriesSplit

# StratifiedKFold: mantiene proporción de clases en cada fold
skf = StratifiedKFold(n_splits=5)

# LeaveOneOut: máximo realismo pero computacionalmente costoso
loo = LeaveOneOut()

# TimeSeriesSplit: para datos temporales (respeta orden temporal)
tscv = TimeSeriesSplit(n_splits=5)
```

---

### 6️⃣ FASE 6: SELECCIÓN Y ENTRENAMIENTO DE MODELOS

#### 6.1 Concepto General
Elegir el modelo correcto depende del problema (clasificación vs regresión) y el tamaño/complejidad de datos.

#### 6.2 Modelos para Clasificación

**Regresión Logística (Baseline)**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Decision Tree**
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)
```

**Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,        # Cantidad de árboles
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1               # Usar todos los núcleos
)
model.fit(X_train, y_train)
```

**Support Vector Machine**
```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)
```

**Gradient Boosting**
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
```

#### 6.3 Modelos para Regresión

**Regresión Lineal (Baseline)**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
```

**XGBoost (Recomendado para competiciones)**
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
```

---

### 7️⃣ FASE 7: EVALUACIÓN Y MÉTRICAS

#### 7.1 Métricas para Clasificación

**Matriz de Confusión:**
```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[TN  FP]
#  [FN  TP]]
```

**Exactitud (Accuracy)**
```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
# Fórmula: (TP + TN) / (TP + TN + FP + FN)
# ⚠️ Engañosa si datos desbalanceados
```

**Precisión (Precision)**
```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
# Fórmula: TP / (TP + FP)
# ¿De las predicciones positivas, cuántas fueron correctas?
```

**Recall (Sensibilidad)**
```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
# Fórmula: TP / (TP + FN)
# ¿Cuántos casos positivos reales detectó?
```

**F1-Score**
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
# Fórmula: 2 * (precision * recall) / (precision + recall)
# Balance entre precision y recall
```

**ROC-AUC**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

**Reporte completo:**
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Clase_0', 'Clase_1']))
```

#### 7.2 Métricas para Regresión

**Mean Absolute Error (MAE)**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
# Promedio de errores absolutos
# Unidades: mismas que la variable target
```

**Mean Squared Error (MSE)**
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
# Penaliza outliers más (cuadrado del error)
```

**Root Mean Squared Error (RMSE)**
```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Raíz cuadrada del MSE
# Interprete más fácil (unidades como target)
```

**R² Score (Coeficiente de Determinación)**
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
# Rango: 0 a 1 (en datos de test puede ser negativo)
# Interpretación: % de varianza explicada
# R² = 0.85 → explica 85% de la varianza
```

**Ejemplo integrado:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluar_modelo_regresion(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

evaluar_modelo_regresion(y_test, y_pred)
```

---

### 8️⃣ FASE 8: AJUSTE FINO Y OPTIMIZACIÓN DE HIPERPARÁMETROS

#### 8.1 Concepto General
Los hiperparámetros son configuraciones del modelo que se establecen **antes** del entrenamiento. Optimizarlos mejora significativamente el rendimiento.

#### 8.2 Grid Search (Búsqueda Exhaustiva)

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Definir grid de parámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Crear GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Entrenar
grid_search.fit(X_train, y_train)

# Resultados
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor CV score: {grid_search.best_score_:.4f}")

# Usar mejor modelo
best_model = grid_search.best_estimator_
```

#### 8.3 Randomized Search (Búsqueda Aleatoria)

```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

param_distributions = {
    'n_estimators': stats.randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': stats.randint(2, 20),
    'learning_rate': stats.uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,  # Número de combinaciones a probar
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Mejores parámetros: {random_search.best_params_}")
```

#### 8.4 Early Stopping (para modelos iterativos)

```python
import xgboost as xgb

model = xgb.XGBClassifier(random_state=42)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)
```

---

### 9️⃣ FASE 9: ANÁLISIS DE RESULTADOS Y DIAGNOSTICO

#### 9.1 Feature Importance (Importancia de Variables)

```python
# Para Random Forest / Gradient Boosting
feature_importance = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Visualizar
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importancia')
plt.show()
```

#### 9.2 Análisis de Residuos (Regresión)

```python
residuos = y_test - y_pred

plt.figure(figsize=(12, 4))

# Gráfico 1: Residuos vs predichos
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuos)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores predichos')
plt.ylabel('Residuos')

# Gráfico 2: Distribución de residuos
plt.subplot(1, 2, 2)
plt.hist(residuos, bins=30, edgecolor='black')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')

plt.show()
```

#### 9.3 Matriz de Confusión Visualizada (Clasificación)

```python
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

---

### 🔟 FASE 10: RECOMENDACIONES Y PRESENTACIÓN DE RESULTADOS

#### 10.1 Estructura de Recomendaciones

**1. Resumen Ejecutivo**
```
- Objetivo del análisis
- Principal hallazgo/insight
- Recomendación principal
```

**2. Análisis de Resultados del Modelo**
```python
# Crear reporte
reporte = f"""
MODELO: {model.__class__.__name__}

RENDIMIENTO EN TEST:
- Exactitud: {accuracy:.3f}
- Precisión: {precision:.3f}
- Recall: {recall:.3f}
- F1-Score: {f1:.3f}

VARIABLES MÁS IMPORTANTES:
{importance_df.head(5).to_string()}

CONCLUSIONES:
- El modelo explica [X]% de la varianza
- Las variables más influyentes son [Y]
- Recomendación: [Z]
"""
print(reporte)
```

**3. Recomendaciones Basadas en Datos**
```python
# Análisis de segmentos
segmentos = df.groupby('categoria').agg({
    'conversión': 'mean',
    'inversión': 'sum',
    'clientes': 'count'
}).sort_values('conversión', ascending=False)

print("RECOMENDACIONES POR SEGMENTO:")
for idx, row in segmentos.iterrows():
    print(f"\n{idx}: Conversión {row['conversión']:.1%}")
    print(f"  → Aumentar inversión en este segmento")
```

#### 10.2 Guardar Modelo para Producción

```python
import pickle
import joblib

# Guardar modelo
joblib.dump(model, 'modelo_produccion.pkl')

# Cargar modelo
model_cargado = joblib.load('modelo_produccion.pkl')

# Hacer predicciones
nuevas_predicciones = model_cargado.predict(nuevos_datos)
```

---

## 📈 FLUJO COMPLETO: EJEMPLO INTEGRADO

```python
# ============================================
# PASO 1: CARGAR DATOS
# ============================================
df = pd.read_csv('datos_clientes.csv')
print(f"Dataset shape: {df.shape}")

# ============================================
# PASO 2: EXPLORACIÓN Y LIMPIEZA
# ============================================
print(df.info())
print(df.describe())
print(f"Valores faltantes: {df.isnull().sum().sum()}")

# Eliminar duplicados
df = df.drop_duplicates()

# Manejar valores faltantes
df['edad'] = df['edad'].fillna(df['edad'].median())
df['ciudad'] = df['ciudad'].fillna('Desconocida')

# ============================================
# PASO 3: FEATURE ENGINEERING & CODIFICACIÓN
# ============================================
# One-hot encoding
df = pd.get_dummies(df, columns=['ciudad'], drop_first=True)

# Feature nuevo
df['edad_grupo'] = pd.cut(df['edad'], bins=[0, 30, 50, 100], 
                          labels=['joven', 'adulto', 'senior'])
df['edad_grupo'] = df['edad_grupo'].map({'joven': 0, 'adulto': 1, 'senior': 2})

# ============================================
# PASO 4: PREPARAR FEATURES Y TARGET
# ============================================
X = df.drop('compra', axis=1)  # Características
y = df['compra']                # Target

# ============================================
# PASO 5: DIVISIÓN Y ESCALADO
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# PASO 6: ENTRENAR MODELO
# ============================================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# ============================================
# PASO 7: PREDICCIONES Y EVALUACIÓN
# ============================================
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# ============================================
# PASO 8: VISUALIZACIÓN Y RECOMENDACIONES
# ============================================
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 características más importantes:")
print(feature_importance.head())

# ============================================
# PASO 9: GUARDAR MODELO
# ============================================
joblib.dump(model, 'modelo_clasificacion.pkl')
print("Modelo guardado exitosamente")
```

---

## 💡 TIPS CLAVE PARA ENTREVISTA

### ✅ LO QUE ESPERA EL HIRING MANAGER

1. **Comprensión integral del flujo** → Desde datos crudos hasta insights accionables
2. **Justificación de decisiones** → "¿Por qué eligió StandardScaler?"
3. **Manejo de casos edge** → Datos desbalanceados, valores faltantes, outliers
4. **Métrica correcta** → No usar accuracy si datos desbalanceados
5. **Validación apropiada** → Cross-validation, no solo train/test
6. **Reproductibilidad** → `random_state=42` en todos los modelos

### ❌ ERRORES COMUNES

- **Data leakage**: Escalar todo antes de dividir
- **Métrica equivocada**: Accuracy en datos desbalanceados
- **Overfitting**: No revisar rendimiento en test
- **No normalizar features** → Algoritmos basados en distancia fallan
- **Ignorar valores faltantes** → Resultados sesgados

---

## 📚 COMANDOS PYTHON FRECUENTES EN ENTREVISTAS

```python
# Cargar datos
pd.read_csv(), pd.read_sql(), pd.read_excel()

# Exploración
df.head(), df.info(), df.describe(), df.isnull().sum()

# Limpieza
df.dropna(), df.fillna(), df.drop_duplicates()

# Transformación
pd.get_dummies(), LabelEncoder(), StandardScaler(), MinMaxScaler()

# Modelado
train_test_split(), KFold(), cross_val_score()

# Evaluación
accuracy_score(), precision_score(), recall_score(), f1_score()
mean_squared_error(), r2_score()

# Optimización
GridSearchCV(), RandomizedSearchCV()

# Guardado
joblib.dump(), joblib.load()
```

---

## 🎓 CONCLUSIÓN

El dominio del pipeline de datos completo es esencial. Los hiring managers evalúan:

1. ¿Entiende el flujo de extremo a extremo?
2. ¿Justifica sus decisiones técnicas?
3. ¿Evita errores comunes (data leakage, overfitting)?
4. ¿Selecciona métricas apropiadas?
5. ¿Comunica resultados claramente?

**Estudie este material, practique con datasets reales en Kaggle, y esté listo para discutir cada decisión que toma durante el proceso.**

---

**Última actualización**: Diciembre 2025
**Versión**: 1.0