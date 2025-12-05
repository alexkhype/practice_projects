# ⚡ REFERENCIA RÁPIDA: COMANDOS PYTHON MÁS UTILIZADOS EN ENTREVISTAS

## Versión 1.0 | Cheatsheet para Entrevista

---

## 📚 LIBRERÍAS ESENCIALES

```python
import pandas as pd                    # Manipulación de datos
import numpy as np                     # Operaciones numéricas
import matplotlib.pyplot as plt        # Visualización básica
import seaborn as sns                  # Visualización avanzada

# Machine Learning
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)
```

---

## 🔴 CARGA DE DATOS

```python
# CSV
df = pd.read_csv('archivo.csv')

# Excel
df = pd.read_excel('archivo.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('archivo.json')

# SQL
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://user:pass@localhost/db')
df = pd.read_sql('SELECT * FROM tabla', con=engine)

# Con parámetros útiles
df = pd.read_csv('datos.csv', sep=',', encoding='utf-8', nrows=1000)
```

---

## 🔍 EXPLORACIÓN RÁPIDA

```python
df.head()               # Primeras 5 filas
df.tail()               # Últimas 5 filas
df.info()               # Tipos y no-nulos
df.describe()           # Estadísticas
df.shape                # Dimensiones (filas, columnas)
df.dtypes               # Tipos por columna
df.columns              # Nombres de columnas
df.isnull().sum()       # Conteo de NaN
df.duplicated().sum()   # Conteo de duplicados
df['col'].value_counts() # Frecuencias
```

---

## 🧹 LIMPIEZA DE DATOS

### Valores Faltantes
```python
df.isnull().sum()                     # Detectar
df.dropna()                           # Eliminar filas con NaN
df.fillna(df.mean())                  # Llenar con media
df['col'].fillna(df['col'].median())  # Mediana de columna
df['col'].fillna('Desconocido')       # Valor específico
df.fillna(method='ffill')             # Forward fill (series temporales)
```

### Duplicados
```python
df.drop_duplicates()                  # Eliminar filas duplicadas
df.drop_duplicates(subset=['col1'])   # Considerar columnas específicas
df.duplicated().sum()                 # Contar duplicados
```

### Outliers (IQR)
```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df_clean = df[(df['col'] >= lower) & (df['col'] <= upper)]
```

### Tipos de Datos
```python
df['col'] = pd.to_numeric(df['col'], errors='coerce')
df['col'] = df['col'].astype('int64')
df['fecha'] = pd.to_datetime(df['fecha'])
df['col'] = df['col'].astype('category')
```

### Strings
```python
df['col'] = df['col'].str.strip()              # Eliminar espacios
df['col'] = df['col'].str.lower()              # Minúsculas
df['col'] = df['col'].str.replace('old', 'new')
df['col'] = df['col'].str.contains('pattern')  # Buscar patrón
```

---

## 🔢 CODIFICACIÓN DE VARIABLES

### Label Encoding (Ordinales)
```python
# Manual (cuando hay orden: Low < Medium < High)
mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['col_encoded'] = df['col'].map(mapping)

# Automático (orden alfabético)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'])
```

### One-Hot Encoding (Nominales)
```python
# Pandas
df_encoded = pd.get_dummies(df, columns=['col'], drop_first=True)

# Sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
X_encoded = ohe.fit_transform(df[['col']])
```

---

## ⚖️ ESCALADO Y NORMALIZACIÓN

### StandardScaler (Z-score)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transform en test
```

### MinMaxScaler (0-1)
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### RobustScaler (ante outliers)
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

## ✂️ DIVISIÓN TRAIN/TEST

```python
# Básica
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Con stratify (clasificación desbalanceada)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# K-Fold Cross-Validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    # Entrenar y evaluar aquí
```

---

## 🤖 MODELOS COMUNES

### Clasificación

**Logistic Regression (Baseline)**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
```

**Gradient Boosting**
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
)
model.fit(X_train, y_train)
```

**SVM**
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train, y_train)
```

### Regresión

**Linear Regression**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)
```

---

## 📊 EVALUACIÓN CLASIFICACIÓN

```python
# Predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

# Precisión
from sklearn.metrics import precision_score
prec = precision_score(y_test, y_pred)

# Recall
from sklearn.metrics import recall_score
rec = recall_score(y_test, y_pred)

# F1-Score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

# ROC-AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred_proba)

# Reporte completo
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

---

## 📈 EVALUACIÓN REGRESIÓN

```python
# Predicciones
y_pred = model.predict(X_test)

# MAE (Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)

# MSE (Mean Squared Error)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# R² Score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

# Combo print
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
```

---

## 🔧 VALIDACIÓN CRUZADA

```python
# cross_val_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Scoring options
# Clasificación: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
# Regresión: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
```

---

## 🎯 OPTIMIZACIÓN DE HIPERPARÁMETROS

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
best_model = grid.best_estimator_
```

### Random Search
```python
from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(
    model, param_grid, n_iter=20, cv=5, scoring='f1', n_jobs=-1, random_state=42
)
random.fit(X_train, y_train)
```

---

## 🌳 FEATURE IMPORTANCE

```python
# Después de entrenar Random Forest o Gradient Boosting
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualizar
import matplotlib.pyplot as plt
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importancia')
plt.show()

# Top 5
print(feature_importance.head())
```

---

## 💾 GUARDAR Y CARGAR MODELOS

```python
import joblib

# Guardar
joblib.dump(model, 'modelo.pkl')

# Cargar
model_cargado = joblib.load('modelo.pkl')

# Predicciones con modelo cargado
y_pred = model_cargado.predict(X_new)
```

---

## 🚀 PIPELINE (RECOMENDADO)

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Entrenar
pipeline.fit(X_train, y_train)

# Predecir (scaler se aplica automáticamente)
y_pred = pipeline.predict(X_test)

# Con GridSearch
param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)
```

---

## 📋 FEATURE ENGINEERING COMÚN

```python
# Variables polinómicas
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

# Variables de interacción
df['X1_x_X2'] = df['X1'] * df['X2']

# Variables temporales
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['dia_semana'] = df['fecha'].dt.dayofweek

# Variables logarítmicas
df['log_precio'] = np.log(df['precio'] + 1)

# Variables binarias
df['es_adulto'] = (df['edad'] >= 18).astype(int)

# Variables de agregación
grupo_stats = df.groupby('categoria')['precio'].agg(['mean', 'std']).reset_index()
df = df.merge(grupo_stats, on='categoria', how='left')
```

---

## 🎨 VISUALIZACIÓN RÁPIDA

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma
plt.hist(df['col'])

# Scatter
plt.scatter(df['col1'], df['col2'])

# Box plot
sns.boxplot(data=df, x='categoria', y='valor')

# Heatmap correlación
sns.heatmap(df.corr(), annot=True)

# Distribución
sns.histplot(df['col'], kde=True)

# Mostrar
plt.show()
```

---

## ✅ CHECKLIST ANTES DE ENTREGAR

```python
# ✓ Datos limpios (sin NaN, duplicados, outliers tratados)
# ✓ Features codificados correctamente
# ✓ Escalado aplicado
# ✓ Train/test split apropiado
# ✓ Cross-validation verificada
# ✓ Métrica correcta para el problema
# ✓ Random state fijado (reproducibilidad)
# ✓ Test performance verificado (no overfitting)
# ✓ Feature importance analizado
# ✓ Modelo guardado y puede ser cargado
```

---

## 💡 TIPS FINALES

- Siempre verificar proporciones de clases con `value_counts()`
- Nunca escalar antes de dividir train/test
- Usar `stratify=y` si datos desbalanceados
- Comparar train vs test performance
- Documentar por qué cada decisión
- Ser consistente con `random_state`
- Probar múltiples modelos, no solo uno

---

**Última actualización**: Diciembre 2025
**Versión**: 1.0