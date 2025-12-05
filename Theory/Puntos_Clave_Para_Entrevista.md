# 🎤 PUNTOS CLAVE PARA DISCUTIR EN ENTREVISTA TÉCNICA

## Versión 1.0 | Talking Points para Hiring Managers

---

## 📝 INTRODUCCIÓN

Cuando el entrevistador pregunte "Cuéntame cómo abordarías un análisis de datos", estos son los puntos clave que debes cubrir. El objetivo es demostrar **pensamiento sistemático, justificación técnica, y evitar errores comunes**.

---

## 1️⃣ FASE DE ENTENDIMIENTO Y EXPLORACIÓN

### Preguntas que Debes Hacer (o Demostrar que Entiendes)

**❌ EVITAR decir:** "Empezaría a cargar datos y hacer un modelo"

**✅ BIEN decir:**
- "Primero necesito entender cuál es el objetivo de negocio"
- "¿Qué métrica es más importante: precisión o recall?"
- "¿Qué tamaño tiene el dataset aproximadamente?"
- "¿Hay desbalance de clases o datos faltantes conocidos?"

### Puntos de Demostración
```
✓ Exploratory Data Analysis (EDA) es fundamental
✓ Entender el contexto evita errores costosos
✓ Pregunta por la métrica de éxito ANTES de modelar
✓ Verificar balanceo de clases en clasificación
```

### Comandos a Mencionar
```python
df.shape              # ¿Cuántas muestras y features?
df.info()             # ¿Qué tipos de datos?
df.describe()         # ¿Cuál es la distribución?
df['target'].value_counts()  # ¿Está balanceado?
```

---

## 2️⃣ LIMPIEZA Y PREPARACIÓN

### Concepto Clave
"*El 80% del trabajo en machine learning es preparación de datos, no modelado*"

### Errores Comunes a EVITAR
```
❌ Ignorar valores faltantes
❌ No investigar outliers antes de eliminarlos
❌ Usar media para imputar si datos tienen outliers
❌ Mantener duplicados en el dataset
❌ No validar consistencia de datos
```

### Lo que Debes Demostrar
```python
# 1. Diagnosticar el problema
print(f"Faltantes: {df.isnull().sum().sum()}")
print(f"Duplicados: {df.duplicated().sum().sum()}")

# 2. Aplicar solución apropiada
# Para valores faltantes: mediana (si outliers), media (si normal)
df['precio'] = df['precio'].fillna(df['precio'].median())

# Para outliers: IQR es estándar
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['col'] >= Q1 - 1.5*IQR) & (df['col'] <= Q3 + 1.5*IQR)]

# 3. Verificar resultado
assert df.isnull().sum().sum() == 0
```

### Puntos de Conversación
- "Mediana es más robusta que media si hay outliers"
- "IQR es el método estándar para detectar outliers"
- "Si >30% de faltantes, preferible eliminar columna"
- "Nunca asumir un valor: siempre investigar causas"

---

## 3️⃣ CODIFICACIÓN Y FEATURE ENGINEERING

### Decisiones Clave

**Pregunta: ¿Label Encoding vs One-Hot Encoding?**

**✅ Respuesta Correcta:**
- "Depende si la variable es **ordinaria** (tiene orden) o **nominal** (no tiene orden)"
- "Si Low < Medium < High → Label Encoding"
- "Si Madrid, Barcelona, Valencia → One-Hot Encoding"

```python
# ORDINARIA: Label Encoding
df['nivel'] = df['nivel'].map({'Low': 0, 'Medium': 1, 'High': 2})

# NOMINAL: One-Hot Encoding
df = pd.get_dummies(df, columns=['ciudad'], drop_first=True)
```

### Feature Engineering - Demuestra Pensamiento Crítico

**❌ MALO:** "No crearía nuevas features, solo uso las que vienen"

**✅ BIEN:**
- "Identificaría features que tengan lógica de negocio"
- "Crearía interacciones si parecen relevantes"
- "Para series temporales, extraería componentes (mes, día_semana)"

```python
# Ejemplo de pensamiento estratégico:
# Si predices compra en e-commerce:
df['es_fin_semana'] = (df['fecha'].dt.dayofweek >= 5).astype(int)
df['gasto_promedio_categoria'] = df.groupby('categoria')['monto'].transform('mean')
df['es_cliente_frecuente'] = (df.groupby('cliente')['compra'].transform('count') > 10).astype(int)
```

### Puntos de Conversación
- "One-hot encoding para nominales evita relaciones falsas"
- "Features de agregación agregan contexto importante"
- "Siempre justificar: ¿por qué es relevante esta feature?"

---

## 4️⃣ ESCALADO Y NORMALIZACIÓN

### Pregunta Típica: "¿Siempre necesitas escalar?"

**✅ Respuesta Correcta:**
```
Depende del algoritmo:

✓ NECESITA: Regresión Lineal, Logística, SVM, KNN, Redes Neuronales
✗ NO NECESITA: Decision Trees, Random Forest, XGBoost
```

### Error Crítico a Evitar

**❌ NUNCA hacer esto:**
```python
# ❌ MAL: Scaler.fit en TODO el dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Incluye test data!
X_train, X_test = train_test_split(X_scaled)  # Data leakage!

# ✅ BIEN: Fit solo en train
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transform
```

### Puntos de Conversación
- "Data leakage es uno de los errores más comunes y costosos"
- "StandardScaler: media=0, std=1 (para mayoría de casos)"
- "MinMaxScaler: rango [0,1] (si necesitas rango específico)"
- "RobustScaler: menos sensible a outliers"

---

## 5️⃣ DIVISIÓN DE DATOS Y VALIDACIÓN

### División Train/Test: ¿Qué Es lo Correcto?

**❌ MALO:**
```python
# Dividir sin stratify en clasificación desbalanceada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**✅ BIEN:**
```python
# Usar stratify si es clasificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Validación Cruzada: Por Qué Es Importante

**Debes explicar:**
- "Train/test split es un snapshot aleatorio"
- "K-Fold CV prueba en múltiples splits"
- "Reduce sesgo y da estimado más confiable"

```python
from sklearn.model_selection import cross_val_score

# En lugar de:
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Un score

# Hacer:
scores = cross_val_score(model, X, y, cv=5)
print(f"Scores: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Puntos de Conversación
- "Stratify mantiene proporciones de clases"
- "CV proporciona estimado más estable"
- "Típicamente 5 o 10 folds"
- "Siempre usar random_state para reproducibilidad"

---

## 6️⃣ SELECCIÓN Y ENTRENAMIENTO DE MODELOS

### Pregunta: "¿Cuál modelo elegirías?"

**✅ Respuesta Estructura:**
```
1. Empezar con modelo simple (baseline)
2. Si rendimiento insuficiente, aumentar complejidad
3. Siempre comparar múltiples modelos
4. No seleccionar basado solo en train score
```

### Baseline Recomendado
```python
# Clasificación: Logistic Regression
# Regresión: Linear Regression
# Estos son rápidos y dan referencia

from sklearn.linear_model import LogisticRegression
baseline = LogisticRegression(random_state=42)
baseline.fit(X_train, y_train)
```

### Modelos a Probar Progresivamente
```
1er intento:  Logistic Regression / Linear Regression
2do intento:  Random Forest
3er intento:  XGBoost / Gradient Boosting
4to intento:  Ensemble (voting classifier)
```

### Puntos de Conversación
- "Random Forest es más robusto que Decision Tree"
- "XGBoost suele dar mejores resultados pero más complejo"
- "No elegir por test score: verificar cross-validation"
- "Interpretabilidad importa: RF mejor que NN para explicabilidad"

---

## 7️⃣ EVALUACIÓN Y MÉTRICAS

### Pregunta Crítica: "¿Qué métrica usarías?"

**❌ NUNCA responder:** "Accuracy siempre"

**✅ BIEN responder:**
```
Depende del problema:

• FRAUDE: Recall (no perder fraudes) o F1
• EMAIL SPAM: Precisión (no falsos positivos)
• DATOS DESBALANCEADOS: F1-Score o ROC-AUC
• DATOS BALANCEADOS: Accuracy es válida
• DATOS TEMPORALES: MAE/RMSE y R²
```

### Justificación en Entrevista

```python
# Si el cliente dice: "95% accuracy"
# ✅ DEBES preguntar:
print(f"Accuracy: {acc:.3f}")
print(f"Pero... ¿cuál es Precision? {prec:.3f}")
print(f"¿Y Recall? {rec:.3f}")
print(f"F1-Score? {f1:.3f}")
print(f"ROC-AUC? {auc:.3f}")

# Porque 95% accuracy en dataset 95% clase A = modelo trivial
```

### Métricas por Escenario

**Clasificación Desbalanceada (fraude, enfermedad):**
- Primary: F1-Score o Recall
- Secondary: Precision, ROC-AUC

**Clasificación Balanceada (normal):**
- Primary: Accuracy, F1-Score
- Secondary: Precision, Recall

**Regresión:**
- Primary: R² (explicación de varianza)
- Secondary: RMSE (error en unidades originales)

### Puntos de Conversación
- "Accuracy engañosa si datos desbalanceados"
- "F1-Score balancea precision y recall"
- "ROC-AUC muestra trade-off a diferentes thresholds"
- "Siempre considerar contexto de negocio"

---

## 8️⃣ DETECCIÓN DE PROBLEMAS: OVERFITTING

### Cómo Identificarlo

**❌ Señal de alerta:**
```
Train accuracy: 99%
Test accuracy:  65%
→ OVERFITTING SEVERO
```

**✅ Lo que debe hacer:**
```python
# 1. Verificar con Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)
if cv_scores.mean() < test_score:
    print("Posible overfitting")

# 2. Reducir complejidad del modelo
# Aumentar regularización
model = DecisionTreeClassifier(
    max_depth=5,           # ← Limitar profundidad
    min_samples_split=10,  # ← Más muestras por split
    min_samples_leaf=5     # ← Más muestras por hoja
)

# 3. Más datos siempre ayuda
# 4. Feature selection (menos features)
```

### Puntos de Conversación
- "Overfitting = memorizar train, fallar en test"
- "CV detecta overfitting mejor que train/test"
- "Regularización reduce complejidad"
- "Menos features = menos overfitting"

---

## 9️⃣ OPTIMIZACIÓN DE HIPERPARÁMETROS

### Pregunta: "¿Cómo ajustarías hiperparámetros?"

**✅ Respuesta Correcta:**
```
Nunca a mano. Usar búsqueda sistemática:
- GridSearchCV: exhaustivo, lento pero completo
- RandomizedSearchCV: aleatorio, rápido, "suficientemente bueno"
```

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',  # ← Métrica correcta
    n_jobs=-1
)

grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
```

### Puntos de Conversación
- "GridSearch es exhaustivo pero lento con muchos parámetros"
- "RandomizedSearch es más rápido, suficientemente bueno"
- "Siempre usar CV dentro de búsqueda"
- "Evaluar en test set al final (no dentro de búsqueda)"

---

## 🔟 INTERPRETABILIDAD Y COMUNICACIÓN

### Feature Importance: Demuestra Comprensión

```python
# Después de entrenar Random Forest
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# ✅ EXPLICAR:
print("Top 3 variables más predictivas:")
for idx, row in feature_imp.head(3).iterrows():
    print(f"• {row['feature']}: {row['importance']:.3f}")
print("\n→ Esto significa que...")
```

### Presentar Resultados Profesionalmente

**❌ MALO:**
```
"El modelo tiene 87% accuracy"
```

**✅ BIEN:**
```
"El modelo explica 87% de la varianza (R²=0.87) y comete 
un error promedio de $5,000 (RMSE). En cross-validation 
de 5 folds obtenemos 85±2%, indicando que generaliza bien.
Las variables más importantes son [X], [Y], [Z]."
```

### Puntos de Conversación
- "Feature importance muestra qué impulsa predicciones"
- "Debe poder explicar cualquier decisión del modelo"
- "Comunicar incertidumbre (±std en CV)"
- "Contexto de negocio siempre"

---

## 🎯 CHECKLIST FINAL: QUE EL ENTREVISTADOR VERIFIQUE

```
Cuando expongas tu solución, el hiring manager evaluará:

□ ¿Entiende el flujo completo de datos?
□ ¿Justifica cada decisión técnica?
□ ¿Evita errores comunes (data leakage, overfitting)?
□ ¿Selecciona métrica apropiada para el problema?
□ ¿Usa validación cruzada correctamente?
□ ¿Documenta random_state para reproducibilidad?
□ ¿Compara múltiples modelos, no solo uno?
□ ¿Verifica que test performance es similar a CV?
□ ¿Analiza feature importance y busca insights?
□ ¿Comunica resultados de forma clara?
```

---

## 📌 ERRORES COSTOSOS: NUNCA COMETERLOS

```
❌ Data Leakage: Escalar antes de dividir train/test
❌ Métrica Equivocada: Usar accuracy en datos desbalanceados
❌ Overfitting No Detectado: Solo revisar train score
❌ No Usar CV: Una sola split de train/test
❌ Random State Sin Fijar: Resultados no reproducibles
❌ Ignorar Faltantes: Basura entra, basura sale
❌ No Balancear Clases: Modelos sesgados
❌ Feature Sin Justificación: "Lo puse porque sí"
❌ Modelo Complejo Sin Razón: Occam's Razor aplica
❌ No Documentar: Código sin comentarios ni razonamiento
```

---

## 🎓 RESPUESTA MODELO A "CUÉNTAME TU ENFOQUE"

```
"Primero, entendería el objetivo de negocio y qué métrica 
importa (precisión, recall, o R²).

Luego, haría EDA: explorar datos, verificar faltantes, 
duplicados, outliers, balanceo de clases.

La limpieza es crucial: medianas para imputar, IQR para 
outliers, validar consistencia.

Después, encoding: one-hot para nominales, label para ordinales.
Feature engineering estratégico, no aleatorio.

Escalaría solo si es necesario (depende del algoritmo), 
aplicando fit solo en training para evitar data leakage.

Dividiría con stratify si clasificación, usaría K-Fold CV 
para validación robusta.

Probaría modelos desde simple (baseline) a complejo (ensemble), 
usando la métrica correcta. Nunca basarme en train score.

Optimizaría hiperparámetros con GridSearchCV en CV.

Finalmente, analizaría feature importance, buscaría insights, 
y comunicaría resultados con contexto de negocio.

Random state fijado en todo. Reproducibilidad garantizada."
```

---

**Última actualización**: Diciembre 2025
**Versión**: 1.0
**Tiempo de lectura**: 30 minutos