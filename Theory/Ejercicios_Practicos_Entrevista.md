# 🎯 EJERCICIOS PRÁCTICOS: ENTREVISTA TÉCNICA DATA ANALYST/SCIENTIST

## Versión 1.0 | Scenarios de Entrevista Real

---

## 📌 INTRODUCCIÓN

Esta sección contiene **ejercicios reales que preguntan en entrevistas técnicas**. Cada ejercicio te prepara para situaciones concretas que encontrarás en el proceso de selección.

---

## EJERCICIO 1: ANÁLISIS COMPLETO DE DATASET DESBALANCEADO

### Escenario
*"Tienes un dataset de detección de fraude. El 98% son transacciones legales y 2% fraudulentas. ¿Cómo evaluarías y entrenarías un modelo?"*

### Datos de Inicio
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Simular dataset desbalanceado
np.random.seed(42)
n_samples = 10000
X = np.random.randn(n_samples, 10)
y = np.random.binomial(1, 0.02, n_samples)  # 2% positivos

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

print(f"Distribución de clases:\n{df['target'].value_counts()}")
print(f"Proporción: {df['target'].mean():.2%} fraude")
```

### Respuesta Esperada

```python
# ❌ MAL: Usar accuracy
accuracy = accuracy_score(y_test, y_pred)  # Sería 98% aunque prediga todo como legítimo

# ✅ BIEN: Usar métricas apropiadas
print("=== EVALUACIÓN ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# ✅ BIEN: Usar class_weight para balancear
model = RandomForestClassifier(
    class_weight='balanced',      # Penaliza más los errores en clase minoritaria
    random_state=42
)

# ✅ BIEN: Usar stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Mantiene proporciones en ambos sets
)
```

### Puntos a Justificar en la Entrevista
- "Usar accuracy es engañoso en datos desbalanceados"
- "F1-Score o ROC-AUC son métricas más apropiadas"
- "Stratified split asegura que ambos sets tienen la misma proporción de clases"
- "class_weight='balanced' penaliza errores en la clase minoritaria"

---

## EJERCICIO 2: MANEJO DE VALORES FALTANTES ESTRATÉGICO

### Escenario
*"Dataset de propiedades inmobiliarias con valores faltantes. ¿Cuál es tu estrategia de imputación?"*

### Datos de Inicio
```python
df = pd.DataFrame({
    'precio': [250000, 300000, np.nan, 450000, 500000],
    'area': [100, np.nan, 150, 200, 250],
    'habitaciones': [3, 4, 5, np.nan, 5],
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', np.nan, 'Valencia'],
    'antiguedad': [5, 10, np.nan, 20, 15]
})

print("Datos faltantes:")
print(df.isnull().sum())
```

### Respuesta Esperada

```python
# Analizar patrón de faltantes
print("\n=== ANÁLISIS DE FALTANTES ===")
print(f"Porcentaje por columna:")
for col in df.columns:
    missing_pct = df[col].isnull().sum() / len(df) * 100
    print(f"{col}: {missing_pct:.1f}%")

# Estrategia diferenciada por tipo de variable

# 1. VARIABLES NUMÉRICAS: Usar mediana (robusta ante outliers)
df['precio'].fillna(df['precio'].median(), inplace=True)
df['area'].fillna(df['area'].median(), inplace=True)
df['habitaciones'].fillna(df['habitaciones'].median(), inplace=True)

# 2. VARIABLES CATEGÓRICAS: Usar moda
df['ciudad'].fillna(df['ciudad'].mode()[0], inplace=True)

# 3. VARIABLES CON LÓGICA TEMPORAL: Forward/Backward fill
df['antiguedad'].fillna(method='ffill', inplace=True)

# ALTERNATIVA: Si más del 30% faltante, eliminar columna
for col in df.columns:
    if df[col].isnull().sum() / len(df) > 0.3:
        print(f"Eliminando {col} (demasiados faltantes)")
        df.drop(col, axis=1, inplace=True)

print("\nDataset limpio:")
print(df.isnull().sum().sum() == 0)  # Verificar que no hay NaN
```

### Puntos a Justificar
- "Mediana es más robusta que media ante outliers"
- "Moda para variables categóricas mantiene la distribución"
- "Si >30% faltante, mejor eliminar que imputar"
- "Nunca usar valores futuros para imputar valores pasados"

---

## EJERCICIO 3: DETECCIÓN Y TRATAMIENTO DE OUTLIERS

### Escenario
*"Dataset de salarios con algunos outliers extremos. ¿Cómo los identificarías y trataría?"*

### Datos de Inicio
```python
np.random.seed(42)
salarios = np.concatenate([
    np.random.normal(30000, 5000, 95),    # 95 salarios normales
    np.array([150000, 200000, 500000, 1000000, 2000000])  # 5 outliers
])

df = pd.DataFrame({'salario': salarios})
print(f"Media: {df['salario'].mean():.2f}")
print(f"Mediana: {df['salario'].median():.2f}")
```

### Respuesta Esperada

```python
# MÉTODO 1: IQR (Recomendado)
Q1 = df['salario'].quantile(0.25)
Q3 = df['salario'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['salario'] < lower_bound) | (df['salario'] > upper_bound)]
print(f"Outliers detectados (IQR): {len(outliers_iqr)}")

# Opción A: Eliminar outliers
df_clean_delete = df[(df['salario'] >= lower_bound) & (df['salario'] <= upper_bound)]

# Opción B: Reemplazar con límites
df_clean_clip = df.copy()
df_clean_clip['salario'] = df_clean_clip['salario'].clip(lower=lower_bound, upper=upper_bound)

# MÉTODO 2: Z-score (para distribuciones normales)
from scipy import stats
z_scores = np.abs(stats.zscore(df['salario']))
df_clean_zscore = df[z_scores < 3]  # Remover si |z| > 3

# MÉTODO 3: Percentiles (pragmático)
p1 = df['salario'].quantile(0.01)
p99 = df['salario'].quantile(0.99)
df_clean_percentile = df[(df['salario'] >= p1) & (df['salario'] <= p99)]

print(f"\nMuestra original: {len(df)}")
print(f"Tras eliminar con IQR: {len(df_clean_delete)}")
print(f"Tras clipear con IQR: {len(df_clean_clip)}")
print(f"Tras percentiles 1-99: {len(df_clean_percentile)}")
```

### Puntos a Justificar
- "IQR es más robusto que Z-score si datos no son normales"
- "Reemplazar en lugar de eliminar preserva tamaño de dataset"
- "Contexto importa: ¿Son errores o valores reales?"
- "Nunca eliminar sin investigar primero qué representa"

---

## EJERCICIO 4: FEATURE ENGINEERING ESTRATÉGICO

### Escenario
*"Dataset de e-commerce con fechas de compra. Crea features relevantes para predecir gasto."*

### Datos de Inicio
```python
df = pd.DataFrame({
    'fecha_compra': pd.date_range('2024-01-01', periods=100, freq='D'),
    'monto': np.random.gamma(shape=2, scale=50, size=100),
    'categoria': np.random.choice(['Electrónica', 'Ropa', 'Libros'], 100)
})

print(df.head())
```

### Respuesta Esperada

```python
# FEATURES TEMPORALES
df['mes'] = df['fecha_compra'].dt.month
df['dia_semana'] = df['fecha_compra'].dt.dayofweek
df['es_fin_semana'] = (df['fecha_compra'].dt.dayofweek >= 5).astype(int)
df['es_primeros_dias_mes'] = (df['fecha_compra'].dt.day <= 5).astype(int)
df['dias_desde_inicio'] = (df['fecha_compra'] - df['fecha_compra'].min()).dt.days

# FEATURES POR AGREGACIÓN
resumen_categoria = df.groupby('categoria')['monto'].agg(['mean', 'std', 'count']).reset_index()
resumen_categoria.columns = ['categoria', 'gasto_promedio_categoria', 'std_categoria', 'freq_categoria']
df = df.merge(resumen_categoria, on='categoria', how='left')

# FEATURES DE TENDENCIA (rolling)
df = df.sort_values('fecha_compra')
df['gasto_promedio_7d'] = df['monto'].rolling(window=7, min_periods=1).mean()
df['volatilidad_7d'] = df['monto'].rolling(window=7, min_periods=1).std()

# FEATURES DE INTERACCIÓN
df['gasto_x_mes'] = df['monto'] * df['mes']
df['es_ropa_fin_semana'] = ((df['categoria'] == 'Ropa') & (df['es_fin_semana'] == 1)).astype(int)

print("\nFeatures creadas:")
print(df.columns.tolist())
print(df.head())
```

### Puntos a Justificar
- "Features temporales capturan patrones de estacionalidad"
- "Agregaciones por grupo añaden contexto"
- "Features de interacción mejoran la capacidad predictiva"
- "Rolling windows útiles para series temporales"

---

## EJERCICIO 5: VALIDACIÓN CRUZADA Y OVERFITTING

### Escenario
*"Tu modelo tiene 99% accuracy en train pero 65% en test. ¿Qué está pasando y cómo lo arreglarías?"*

### Datos de Inicio
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Dataset pequeño (propenso a overfitting)
X = np.random.randn(100, 20)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo sin regularización
model_sin_reg = DecisionTreeClassifier(random_state=42)
model_sin_reg.fit(X_train, y_train)

print(f"Train accuracy: {model_sin_reg.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model_sin_reg.score(X_test, y_test):.4f}")
print("⚠️ Gran diferencia = OVERFITTING")
```

### Respuesta Esperada

```python
# DIAGNÓSTICO 1: Cross-Validation
cv_scores = cross_val_score(model_sin_reg, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCV Scores: {cv_scores}")
print(f"CV Media: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print("→ Si CV promedio < Test accuracy, hay overfitting")

# SOLUCIÓN 1: Regularización (max_depth)
model_reg = DecisionTreeClassifier(max_depth=5, random_state=42)
model_reg.fit(X_train, y_train)

print(f"\nCon max_depth=5:")
print(f"Train accuracy: {model_reg.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model_reg.score(X_test, y_test):.4f}")
print("→ Diferencia reducida")

# SOLUCIÓN 2: Más features regularización
model_reg2 = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model_reg2.fit(X_train, y_train)

print(f"\nCon múltiples restricciones:")
print(f"Train accuracy: {model_reg2.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model_reg2.score(X_test, y_test):.4f}")

# SOLUCIÓN 3: Ensemble (Random Forest)
from sklearn.ensemble import RandomForestClassifier
model_ensemble = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_ensemble.fit(X_train, y_train)

print(f"\nRandom Forest:")
print(f"Train accuracy: {model_ensemble.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model_ensemble.score(X_test, y_test):.4f}")
print("→ Combinar múltiples modelos reduce overfitting")
```

### Puntos a Justificar
- "Gran diferencia train/test = overfitting"
- "Cross-validation detecta overfitting"
- "Regularización (max_depth, min_samples) reduce complejidad"
- "Ensemble methods generalizan mejor"

---

## EJERCICIO 6: SELECCIÓN DE MÉTRICA CORRECTA

### Escenario
*"Clasificador de emails (spam vs. no-spam). Tu cliente dice: 'No me importa perder spam, pero no puedo tener emails legítimos en carpeta spam'. ¿Cuál métrica optimizarías?"*

### Análisis
```python
# Problema: Email legítimo (1) vs Spam (0)
# Cliente importa: NO falsos positivos (email legítimo predicho como spam)
# Poco importa: Falsos negativos (spam predicho como legítimo)

# MATRIZ DE CONFUSIÓN:
#                 Predicción
#              Legítimo  Spam
# Realidad  L    TP       FP  ← Este error es crítico (cliente insatisfecho)
#           S    FN       TN

# MÉTRICA A OPTIMIZAR: Precisión
# Precisión = TP / (TP + FP)
# "De los emails que predije como legítimos, ¿cuántos realmente lo son?"

from sklearn.metrics import classification_report, confusion_matrix

y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
y_pred = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

print("=== MATRIZ DE CONFUSIÓN ===")
cm = confusion_matrix(y_true, y_pred)
print(f"TP: {cm[1,1]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TN: {cm[0,0]}")

print("\n=== MÉTRICAS ===")
print(classification_report(y_true, y_pred, target_names=['Spam', 'Legítimo']))

precision = cm[1,1] / (cm[1,1] + cm[0,1])
recall = cm[1,1] / (cm[1,1] + cm[1,0])

print(f"\nPrecisión: {precision:.2%} (Pocos FP)")
print(f"Recall: {recall:.2%} (Puede haber FN)")
print("\n→ Para este caso, priorizar PRECISIÓN")
```

### Matriz de Decisión de Métricas
```
CASO                    MÉTRICA IMPORTANTE    RAZÓN
─────────────────────────────────────────────────────────────
Detección fraude        Recall (F1)           No perder fraudes
Diagnóstico médico      Recall                No perder enfermos
Email spam              Precisión             No falsos positivos
Recomendaciones         Precisión             Solo items relevantes
Desbalanceados          F1 o ROC-AUC          Accuracy engañosa
Clases equales          Accuracy              Métrica válida
```

---

## EJERCICIO 7: ANÁLISIS COMPLETO REGRESIÓN

### Escenario
*"Predecir precio de casas. Entrena modelo y presenta resultados."*

### Solución Completa
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# PASO 1: CARGAR DATOS
housing = fetch_california_housing()
X = housing.data
y = housing.target
df = pd.DataFrame(X, columns=housing.feature_names)
df['price'] = y

# PASO 2: EXPLORACIÓN
print("=== EXPLORACIÓN ===")
print(df.describe())
print(f"Valores faltantes: {df.isnull().sum().sum()}")

# PASO 3: PREPARAR
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PASO 4: ESCALAR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PASO 5: ENTRENAR
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# PASO 6: EVALUAR
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

print("\n=== EVALUACIÓN ===")
print(f"Train R²:  {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²:   {r2_score(y_test, y_pred_test):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

# PASO 7: CROSS-VALIDATION
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\nCV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# PASO 8: FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== VARIABLES MÁS IMPORTANTES ===")
print(feature_importance)

# PASO 9: RECOMENDACIONES
print("\n=== CONCLUSIONES ===")
print(f"• El modelo explica {r2_score(y_test, y_pred_test):.1%} de la varianza del precio")
print(f"• Error promedio: ${mean_absolute_error(y_test, y_pred_test):.2f} (en escala 0-5)")
print(f"• Variables más influyentes: {', '.join(feature_importance.head(3)['feature'].tolist())}")
print("• Modelo estable: diferencia train/test < 5%")
```

---

## EJERCICIO 8: PRESENTAR RESULTADOS EN ENTREVISTA

### Estructura Esperada de Respuesta

**Cuando el entrevistador pregunta: "Cuéntame cómo abordarías este problema..."**

```
1. ENTENDER EL PROBLEMA (30 segundos)
   ✓ "Necesitamos clasificar X con objetivo Y"
   ✓ "Métrica a optimizar: [MÉTRICA]"
   ✓ "Tamaño aproximado: X muestras, Y features"

2. ESTRATEGIA DE DATOS (1 minuto)
   ✓ "Primero, carga y exploración"
   ✓ "Buscar valores faltantes, outliers, desbalance"
   ✓ "Si desbalanceado, usar stratified split"

3. PREPARACIÓN (1 minuto)
   ✓ "Encoding de categóricas: one-hot si categorías ≤ 10, target encoding si > 10"
   ✓ "Escalado: StandardScaler para [modelos], MinMaxScaler para [modelos]"
   ✓ "Feature engineering relevante para [caso]"

4. MODELADO (1 minuto)
   ✓ "Empezar con baseline simple (Logistic Regression / Linear Regression)"
   ✓ "Luego probar ensemble (Random Forest, XGBoost)"
   ✓ "Usar cross-validation para evitar overfitting"

5. EVALUACIÓN (1 minuto)
   ✓ "Métricas: [MÉTRICA PRINCIPAL] y [SECUNDARIA]"
   ✓ "Validar en test set, no solo train"
   ✓ "Analizar qué casos falla el modelo"

6. OPTIMIZACIÓN (30 segundos)
   ✓ "GridSearchCV o RandomizedSearchCV"
   ✓ "Ajustar hiperparámetros basado en CV"

7. INSIGHTS (1 minuto)
   ✓ "Feature importance"
   ✓ "Qué variables son más predictivas"
   ✓ "Recomendaciones accionables"
```

---

## 📋 CHECKLIST DE VERIFICACIÓN ANTES DE PRESENTAR

```python
# ✅ ANTES DE CUALQUIER RESPUESTA, VERIFICAR:

checklist = {
    "Carga de datos": "¿Entiendo la estructura y tamaño?",
    "Exploración": "¿Sé qué variables tengo, tipos, faltantes?",
    "Limpieza": "¿Manejé outliers y faltantes apropiadamente?",
    "Encoding": "¿Categorías están codificadas correctamente?",
    "Escalado": "¿Escala es apropiada para el algoritmo?",
    "División": "¿Usé stratified split si es clasificación?",
    "Modelo": "¿Justifiqué por qué este modelo?",
    "Validación": "¿Usé cross-validation, no solo train/test?",
    "Métrica": "¿Métrica alineada con objetivo de negocio?",
    "Overfitting": "¿Diferencia train/test es aceptable?",
    "Feature importance": "¿Qué variables son más predictivas?",
    "Reproducibilidad": "¿Fijé random_state en todos lados?",
}

for aspecto, pregunta in checklist.items():
    print(f"[ ] {aspecto}: {pregunta}")
```

---

## 🎓 COMANDO FINAL: PIPELINE COMPLETO TEMPLATE

```python
# COPIAR Y ADAPTAR PARA CUALQUIER PROBLEMA

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

# Pipeline automatizado
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Hiperparámetros a tunear
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, 15],
    'model__min_samples_split': [2, 5, 10]
}

# Grid Search con CV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Entrenar
grid_search.fit(X_train, y_train)

# Mejores parámetros
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Evaluar
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
```

---

**Última actualización**: Diciembre 2025
**Versión**: 1.0
**Tiempo estimado de práctica**: 15-20 horas