# 📊 TABLA COMPACTA: MODELOS DE PREDICCIÓN - CLASIFICACIÓN Y REGRESIÓN

## Versión 1.0 | Referencia Rápida

---

## 🎯 CLASIFICACIÓN - TABLA RÁPIDA

| Modelo | Librería | Funciona | Parámetros Clave | Métricas Primarias | Cuándo Usarlo |
|--------|----------|----------|-----------------|-------------------|--------------|
| **Logistic Regression** | `sklearn.linear_model` | Hiperplano lineal con función sigmoide | `C`, `penalty`, `max_iter` | Accuracy, Precision, Recall, F1 | Baseline, interpretabilidad |
| **Decision Tree** | `sklearn.tree` | Reglas recursivas if-else, particiones | `max_depth`, `min_samples_split` | Accuracy, F1-Score | Exploración, interpretabilidad |
| **Random Forest** | `sklearn.ensemble` | Múltiples árboles, agregación por votación | `n_estimators`, `max_depth`, `max_features` | Accuracy, F1, ROC-AUC | Propósito general, balance |
| **SVM** | `sklearn.svm` | Hiperplano con margen máximo, kernel trick | `C`, `kernel`, `gamma`, `degree` | Accuracy, F1, ROC-AUC | Datos pequeños, dimensiones altas |
| **KNN** | `sklearn.neighbors` | K vecinos más cercanos, votación mayoría | `n_neighbors`, `weights`, `metric` | Accuracy, F1-Score | Baseline rápido, datos pequeños |
| **Naive Bayes** | `sklearn.naive_bayes` | Probabilidades condicionales, independencia | `var_smoothing` | Accuracy, F1-Score, Precision | Texto/NLP, velocidad máxima |
| **Gradient Boosting** | `sklearn.ensemble` | Árboles secuenciales, cada uno corrige errores | `n_estimators`, `learning_rate`, `max_depth` | F1, ROC-AUC, Accuracy | Precisión crítica, datos medianos |
| **XGBoost** | `xgboost` | GB optimizado: regularización, early stopping | `n_estimators`, `max_depth`, `learning_rate` | F1, ROC-AUC, Accuracy | Competiciones, producción |
| **LightGBM** | `lightgbm` | GB ultrarrápido: leaf-wise, histogramas | `n_estimators`, `num_leaves`, `learning_rate` | F1, ROC-AUC, Accuracy | Datos grandes (>100k), velocidad |
| **CatBoost** | `catboost` | GB optimizado para categorías | `n_estimators`, `max_depth`, `cat_features` | F1, ROC-AUC, Accuracy | Muchas categorías, automatización |
| **Neural Networks** | `sklearn.neural_network` | Capas con activación, backpropagation | `hidden_layer_sizes`, `activation`, `max_iter` | Accuracy, F1, ROC-AUC | Datos enormes, complejidad |

---

## 📈 REGRESIÓN - TABLA RÁPIDA

| Modelo | Librería | Funciona | Parámetros Clave | Métricas Primarias | Cuándo Usarlo |
|--------|----------|----------|-----------------|-------------------|--------------|
| **Linear Regression** | `sklearn.linear_model` | Línea ajustada: minimiza MSE | `fit_intercept` | R², RMSE, MAE | Baseline, relaciones lineales |
| **Decision Tree Regressor** | `sklearn.tree` | Particiones recursivas, promedios en hojas | `max_depth`, `min_samples_split` | R², RMSE, MAE | Exploración, interpretabilidad |
| **Random Forest Regressor** | `sklearn.ensemble` | Múltiples árboles, promedio predicciones | `n_estimators`, `max_depth`, `max_features` | R², RMSE, MAE | Propósito general, balance |
| **SVR** | `sklearn.svm` | ε-tubo de soporte vectorial | `C`, `epsilon`, `kernel`, `gamma` | R², RMSE, MAE | Datos pequeños, kernels complejos |
| **Ridge Regression** | `sklearn.linear_model` | Linear + penalidad L2 en coeficientes | `alpha` | R², RMSE, MAE | Multicolinealidad, interpretabilidad |
| **Lasso Regression** | `sklearn.linear_model` | Linear + penalidad L1 en coeficientes | `alpha`, `max_iter` | R², RMSE, MAE | Feature selection, p >> n |
| **Elastic Net** | `sklearn.linear_model` | Linear + L1+L2, Ridge+Lasso | `alpha`, `l1_ratio` | R², RMSE, MAE | Correlacionadas + feature sel. |
| **Gradient Boosting Regressor** | `sklearn.ensemble` | Árboles secuenciales en residuos | `n_estimators`, `learning_rate`, `max_depth` | R², RMSE, MAE | Precisión crítica, datos medianos |
| **XGBoost Regressor** | `xgboost` | GB optimizado: reg., NaN, early stop | `n_estimators`, `max_depth`, `learning_rate` | R², RMSE, MAE | Producción, datasets grandes |
| **LightGBM Regressor** | `lightgbm` | GB ultrarrápido: leaf-wise | `n_estimators`, `num_leaves`, `learning_rate` | R², RMSE, MAE | Datos grandes (>100k), velocidad |
| **CatBoost Regressor** | `catboost` | GB optimizado para categorías | `n_estimators`, `max_depth`, `cat_features` | R², RMSE, MAE | Muchas categorías, automatización |
| **Neural Networks** | `sklearn.neural_network` | Capas con activación | `hidden_layer_sizes`, `activation`, `max_iter` | R², RMSE, MAE | Datos enormes, complejidad |

---

## ⭐ RANKING POR CRITERIO

### TOP 3 VELOCIDAD (Entrenamiento)
1. **Naive Bayes** ⚡⚡⚡⚡⚡
2. **Linear/Logistic Regression** ⚡⚡⚡⚡⚡
3. **KNN** ⚡⚡⚡⚡⚡

### TOP 3 PRECISIÓN (Performance)
1. **XGBoost** 🏆 228 wins
2. **LightGBM** 🏆 242 wins
3. **CatBoost** 🏆 243 wins

### TOP 3 INTERPRETABILIDAD
1. **Linear/Logistic Regression** 📖
2. **Decision Tree** 📖
3. **Naive Bayes** 📖

### TOP 3 ESCALABILIDAD (Datos grandes)
1. **LightGBM** 📊 (>100k muestras)
2. **Naive Bayes** 📊
3. **Neural Networks** 📊

### TOP 3 VERSÁTILES (Usos múltiples)
1. **Random Forest** ✅ (todos los tamaños)
2. **XGBoost** ✅ (general purpose)
3. **Logistic/Linear Regression** ✅ (baseline + interpretabilidad)

---

## 📋 MATRIZ: CUÁL ELEGIR SEGÚN SITUACIÓN

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELECCIÓN DE MODELO                          │
└─────────────────────────────────────────────────────────────────┘

CLASIFICACIÓN:
├─ Datos pequeños (<5k)           → SVM, KNN, Logistic Reg
├─ Datos medianos (5k-1M)         → Random Forest, XGBoost
├─ Datos grandes (>1M)            → LightGBM, Naive Bayes
├─ Necesito interpretabilidad      → Decision Tree, Logistic Reg
├─ Necesito máxima precisión      → CatBoost, XGBoost, LightGBM
├─ Datos desbalanceados           → Random Forest (class_weight)
├─ Muchas categorías              → CatBoost
└─ Texto/NLP                      → Naive Bayes

REGRESIÓN:
├─ Relación lineal conocida       → Linear, Ridge, Lasso
├─ Datos pequeños (<5k)           → SVR, Linear Regression
├─ Datos medianos (5k-1M)         → Random Forest, XGBoost
├─ Datos grandes (>1M)            → LightGBM
├─ Multicolinealidad              → Ridge, Elastic Net
├─ Feature selection automático   → Lasso, Elastic Net
├─ Máxima precisión               → XGBoost, LightGBM, CatBoost
└─ Muchas categorías              → CatBoost
```

---

## 🎯 CHEATSHEET: IMPORTS Y INSTANCIA

```python
# ============ CLASIFICACIÓN ============

# Baseline
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(max_iter=1000, random_state=42)

# Árboles
from sklearn.tree import DecisionTreeClassifier
clf2 = DecisionTreeClassifier(max_depth=10, random_state=42)

from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# SVM y Vecinos
from sklearn.svm import SVC
clf4 = SVC(kernel='rbf', C=1.0, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
clf5 = KNeighborsClassifier(n_neighbors=5)

# Probabilístico
from sklearn.naive_bayes import GaussianNB
clf6 = GaussianNB()

# Boosting
from sklearn.ensemble import GradientBoostingClassifier
clf7 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Especializados
import xgboost as xgb
clf8 = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

import lightgbm as lgb
clf9 = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=42)

import catboost as cb
clf10 = cb.CatBoostClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=0)

# Deep Learning
from sklearn.neural_network import MLPClassifier
clf11 = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)


# ============ REGRESIÓN ============

# Baseline
from sklearn.linear_model import LinearRegression
reg1 = LinearRegression()

# Regularización
from sklearn.linear_model import Ridge, Lasso, ElasticNet
reg2 = Ridge(alpha=1.0, random_state=42)
reg3 = Lasso(alpha=0.1, random_state=42)
reg4 = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# Árboles
from sklearn.tree import DecisionTreeRegressor
reg5 = DecisionTreeRegressor(max_depth=10, random_state=42)

from sklearn.ensemble import RandomForestRegressor
reg6 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# SVM
from sklearn.svm import SVR
reg7 = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Boosting
from sklearn.ensemble import GradientBoostingRegressor
reg8 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Especializados
import xgboost as xgb
reg9 = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

import lightgbm as lgb
reg10 = lgb.LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=42)

import catboost as cb
reg11 = cb.CatBoostRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=0)

# Deep Learning
from sklearn.neural_network import MLPRegressor
reg12 = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
```

---

## 📊 MÉTRICAS DE EVALUACIÓN ESTÁNDAR

### CLASIFICACIÓN

| Métrica | Fórmula | Interpret. | Cuándo Usar |
|---------|---------|-----------|-----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | % correcto | Datos balanceados |
| **Precision** | TP/(TP+FP) | % pred. + correctas | Falsos positivos caros |
| **Recall** | TP/(TP+FN) | % casos + detectados | Falsos negativos caros |
| **F1-Score** | 2*P*R/(P+R) | Balance P y R | Datos desbalanceados |
| **ROC-AUC** | Área bajo curva | Trade-off FP-TP | Ranking, threshold |
| **PR-Curve AUC** | Prec-Recall | Para desbalance severo | Datos muy desbalanceados |

### REGRESIÓN

| Métrica | Fórmula | Interpret. | Cuándo Usar |
|---------|---------|-----------|-----------|
| **MAE** | Σ\|y-ŷ\|/n | Error promedio | Interpretabilidad |
| **MSE** | Σ(y-ŷ)²/n | Error al cuadrado | Optimización |
| **RMSE** | √MSE | Error en unidades | Interpretabilidad |
| **R² Score** | 1 - Σ(y-ŷ)²/Σ(y-ȳ)² | % varianza explicada | Comparación general |
| **MAPE** | Σ\|y-ŷ\|/y*100 | Error porcentual | Series con scale variable |

---

## 🚀 WORKFLOW RECOMENDADO

```
1. EMPEZAR SIMPLE (Baseline)
   ↓
   └─ Clasificación: Logistic Regression
   └─ Regresión: Linear Regression
   └─ Medir performance base

2. PROBAR ENSEMBLE GENERAL
   ↓
   └─ Random Forest (ambos)
   └─ Comparar vs baseline
   └─ Verificar CV stability

3. INTENTAR BOOSTING
   ↓
   └─ Si datos < 100k: XGBoost
   └─ Si datos > 100k: LightGBM
   └─ Si muchas categorías: CatBoost

4. OPTIMIZAR MEJOR MODELO
   ↓
   └─ GridSearchCV o RandomizedSearchCV
   └─ Tuning de hiperparámetros
   └─ Early stopping si disponible

5. VALIDAR Y COMPARAR
   ↓
   └─ Cross-validation
   └─ Test set final
   └─ Feature importance analysis

6. PRODUCCIÓN
   ↓
   └─ Joblib dump del modelo
   └─ Pipeline con scaler
   └─ Monitoreo y actualizaciones
```

---

## 💡 TIPS PRÁCTICOS

### Para Clasificación
- ✓ Siempre usar `stratify=y` en train_test_split
- ✓ Si desbalanceado: verificar ROC-AUC, no solo Accuracy
- ✓ Gradient Boosting > Random Forest en competiciones
- ✓ LightGBM si tiempo es crítico

### Para Regresión
- ✓ Siempre mostrar R² + RMSE
- ✓ Verificar distribución de residuos
- ✓ Ridge si multicolinealidad
- ✓ Lasso si feature selection importante

### General
- ✓ NUNCA escalar antes de dividir
- ✓ Siempre usar `random_state=42`
- ✓ CV antes de test set
- ✓ Feature importance después del training
- ✓ Comparar mínimo 3 modelos

---

## 📚 COMPARATIVA: INVESTIGACIÓN RECIENTE (2024)

Basado en análisis de algoritmos en Kaggle y repositorios académicos:

**Mejor Overall Performance:**
- 🥇 **CatBoost**: 243 wins totales (114 binary, 39 multiclass, 90 regression)
- 🥈 **LightGBM**: 242 wins totales (108 binary, 42 multiclass, 92 regression)
- 🥉 **XGBoost**: 233 wins totales (108 binary, 37 multiclass, 88 regression)

**Mejor Velocidad:**
- ⚡ **LightGBM**: 1-10x más rápido que XGBoost
- ⚡ **Linear Models**: Baseline de velocidad
- ⚡ **Naive Bayes**: Más rápido aún

**Mejor Trade-off Precisión/Velocidad:**
- ⚖️ **LightGBM**: Velocidad + Precisión
- ⚖️ **CatBoost**: Automatización + Precisión
- ⚖️ **Random Forest**: Estabilidad + Performance

---

**Última actualización**: Diciembre 2025
**Versión**: 1.0
**Uso**: Referencia rápida en entrevistas y desarrollo