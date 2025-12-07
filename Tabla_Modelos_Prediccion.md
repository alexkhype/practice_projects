# 📊 TABLA COMPARATIVA: MODELOS DE PREDICCIÓN (CLASIFICACIÓN Y REGRESIÓN)

## Versión 1.0 | Referencia Completa de Modelos ML

---

## 🎯 MODELOS DE CLASIFICACIÓN

### 1. Logistic Regression

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.linear_model import LogisticRegression` |
| **Tipo** | Clasificación (Binaria y Multiclase) |
| **Cómo funciona** | Usa función sigmoide para mapear valores a probabilidad entre 0-1. Encuentra hiperplano que separa clases. Modelo lineal que asume relación lineal entre features y log-odds. |
| **Fórmula** | P(y=1) = 1 / (1 + e^(-z)), donde z = β₀ + β₁x₁ + ... + βₙxₙ |
| **Hiperparámetros clave** | `C` (regularización), `penalty` (L1/L2), `max_iter`, `solver`, `random_state` |
| **Mejor evaluado con** | **Primaria**: Accuracy, Precision, Recall, F1-Score<br>**Secundaria**: ROC-AUC, PR-Curve |
| **Fortalezas** | ✓ Interpretable, ✓ Rápido, ✓ Baseline excelente, ✓ Probabilidades calibradas, ✓ Bajo uso de memoria |
| **Debilidades** | ✗ Asume linealidad, ✗ Mal con datos no separables, ✗ Sensible a outliers, ✗ No captura interacciones automáticas |
| **Mejor para** | Problemas de baseline, interpretabilidad crítica, datasets pequeños, inicio de análisis |
| **Código típico** | `model = LogisticRegression(max_iter=1000, random_state=42)` |

---

### 2. Decision Tree (Árbol de Decisión)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.tree import DecisionTreeClassifier` |
| **Tipo** | Clasificación (Binaria y Multiclase) |
| **Cómo funciona** | Recursivamente particiona feature space usando reglas if-else. En cada nodo, selecciona feature que maximiza ganancia de información (Gini o Entropy). Crea árbol de decisión interpretable. |
| **Criterio de split** | Gini Impurity o Information Gain (Entropy) |
| **Hiperparámetros clave** | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `random_state` |
| **Mejor evaluado con** | **Primaria**: Accuracy, F1-Score<br>**Secundaria**: Precision, Recall, Confusion Matrix |
| **Fortalezas** | ✓ Muy interpretable, ✓ Maneja no-linealidad, ✓ Sin escalado necesario, ✓ Rápido en predicción, ✓ Feature importance claro |
| **Debilidades** | ✗ Propenso a overfitting, ✗ Inestable (pequeños cambios = árbol diferente), ✗ Sesgo hacia features dominantes |
| **Mejor para** | Exploración inicial, interpretabilidad crítica, datos pequeños a medianos |
| **Código típico** | `model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)` |

---

### 3. Random Forest (Clasificación)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.ensemble import RandomForestClassifier` |
| **Tipo** | Clasificación (Binaria y Multiclase) - Ensemble |
| **Cómo funciona** | Crea múltiples decision trees (típicamente 100) usando bootstrap samples. Cada árbol ve subset aleatorio de features. Predica agregando (voting/averaging). Reduce overfitting mediante ensemble. |
| **Agregación** | Mayoría de votos (voting) para clasificación |
| **Hiperparámetros clave** | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `random_state`, `n_jobs` |
| **Mejor evaluado con** | **Primaria**: Accuracy, F1-Score, ROC-AUC<br>**Secundaria**: Precision, Recall, Confusion Matrix |
| **Fortalezas** | ✓ Robusto a overfitting, ✓ Maneja interacciones, ✓ Sin escalado, ✓ Feature importance, ✓ Paralelizable, ✓ Performance consistente |
| **Debilidades** | ✗ Menos interpretable que árbol único, ✗ Más lento que árbol, ✗ Más parámetros a tunear, ✗ Más memoria |
| **Mejor para** | Problemas generales, datasets medianos a grandes, cuando necesitas balance interpretabilidad-performance |
| **Código típico** | `model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)` |

---

### 4. Support Vector Machine (SVM)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.svm import SVC` |
| **Tipo** | Clasificación (Binaria y Multiclase) |
| **Cómo funciona** | Encuentra hiperplano que maximiza margen entre clases. Usa kernel trick para mapear datos a dimensiones superiores. Minimiza error + complejidad. |
| **Kernels disponibles** | `linear`, `rbf` (Gaussian), `poly` (polinomial), `sigmoid` |
| **Hiperparámetros clave** | `C` (regularización), `kernel`, `gamma`, `degree` (si poly), `random_state` |
| **Mejor evaluado con** | **Primaria**: Accuracy, F1-Score, ROC-AUC<br>**Secundaria**: Precision, Recall |
| **Fortalezas** | ✓ Muy efectivo en altas dimensiones, ✓ Versátil (múltiples kernels), ✓ Memoria eficiente, ✓ Buen con datos pequeños |
| **Debilidades** | ✗ Lento con datasets grandes, ✗ Requiere escalado, ✗ Sensible a outliers, ✗ Difícil de interpretar, ✗ Tuning complejo |
| **Mejor para** | Datasets pequeños a medianos, datos de alta dimensión, cuando interpretabilidad no es crítica |
| **Código típico** | `model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)` |

---

### 5. K-Nearest Neighbors (KNN)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.neighbors import KNeighborsClassifier` |
| **Tipo** | Clasificación (Binaria y Multiclase) |
| **Cómo funciona** | Algoritmo lazy (no entrena explícitamente). Para predecir: encuentra K puntos más cercanos en training set. Predice por mayoría de votos de vecinos. Distancia típicamente Euclidiana. |
| **Distancia** | Euclidiana, Manhattan, Minkowski, etc. |
| **Hiperparámetros clave** | `n_neighbors` (K), `weights` (uniform/distance), `metric` (distancia), `algorithm`, `n_jobs` |
| **Mejor evaluado con** | **Primaria**: Accuracy, F1-Score<br>**Secundaria**: Precision, Recall, Confusion Matrix |
| **Fortalezas** | ✓ Simple e intuitivo, ✓ No asume forma lineal, ✓ Rápido en predicción (si índices), ✓ Baseline razonable |
| **Debilidades** | ✗ Lento en entrenamiento (busca vecinos), ✗ Requiere escalado, ✗ Sensible a outliers, ✗ Propenso a overfitting con K pequeño, ✗ Basura entra = basura sale |
| **Mejor para** | Baseline rápido, datasets pequeños, cuando simpleza es importante |
| **Código típico** | `model = KNeighborsClassifier(n_neighbors=5, weights='distance')` |

---

### 6. Naive Bayes

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB` |
| **Tipo** | Clasificación (Binaria y Multiclase) |
| **Cómo funciona** | Aplica Teorema de Bayes asumiendo independencia condicional entre features. P(y\|X) = P(X\|y) * P(y) / P(X). Rápido y eficiente. Asume features independientes (muy simplista). |
| **Variantes** | GaussianNB (continuas), MultinomialNB (conteos), BernoulliNB (binarias) |
| **Hiperparámetros clave** | `var_smoothing` (GaussianNB), muy pocos parámetros |
| **Mejor evaluado con** | **Primaria**: Accuracy, F1-Score<br>**Secundaria**: Precision, Recall, ROC-AUC |
| **Fortalezas** | ✓ Muy rápido, ✓ Bajo uso de memoria, ✓ Funciona bien con texto/NLP, ✓ Pocos datos suficientes |
| **Debilidades** | ✗ Asume independencia (generalmente falso), ✗ Puede ser menos preciso, ✗ Sensible a zero frequency |
| **Mejor para** | Clasificación de texto, spam detection, cuando velocidad es crítica, datasets pequeños |
| **Código típico** | `model = GaussianNB()` |

---

### 7. Gradient Boosting (Clasificación)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.ensemble import GradientBoostingClassifier` |
| **Tipo** | Clasificación (Binaria y Multiclase) - Ensemble |
| **Cómo funciona** | Construye árboles secuencialmente. Cada árbol corrige errores del anterior. Minimiza loss function usando gradient descent. Actualiza predicciones iterativamente. Más lento pero más preciso que Random Forest. |
| **Proceso** | 1. Predecir con árbol 1, 2. Calcular residuos, 3. Entrenar árbol 2 en residuos, 4. Agregar α*predicción_árbol_2 |
| **Hiperparámetros clave** | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `subsample`, `random_state` |
| **Mejor evaluado con** | **Primaria**: ROC-AUC, F1-Score<br>**Secundaria**: Accuracy, Precision, Recall |
| **Fortalezas** | ✓ Muy preciso, ✓ Maneja interacciones, ✓ Sin escalado, ✓ Feature importance, ✓ Regularización built-in |
| **Debilidades** | ✗ Lento en entrenamiento, ✗ Propenso a overfitting, ✗ Más parámetros, ✗ Menos paralelizable |
| **Mejor para** | Competiciones Kaggle, cuando precisión es crítica, datasets medianos |
| **Código típico** | `model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)` |

---

### 8. XGBoost (eXtreme Gradient Boosting)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `import xgboost as xgb` → `xgb.XGBClassifier()` |
| **Tipo** | Clasificación (Binaria y Multiclase) - Gradient Boosting optimizado |
| **Cómo funciona** | Versión optimizada de Gradient Boosting. Más rápido y preciso. Regularización L1/L2 built-in. Maneja valores faltantes automáticamente. Early stopping disponible. |
| **Ventajas sobre GB** | Regularización, paralelización, early stopping, manejo de NaN, fast tree pruning |
| **Hiperparámetros clave** | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `lambda` (L2), `alpha` (L1), `random_state` |
| **Mejor evaluado con** | **Primaria**: ROC-AUC, F1-Score<br>**Secundaria**: Accuracy, Precision, Recall |
| **Fortalezas** | ✓ Muy preciso, ✓ Rápido (GPU support), ✓ Maneja missing values, ✓ Regularización automática, ✓ Early stopping, ✓ Feature importance |
| **Debilidades** | ✗ Overfitting si no se cuida, ✗ Curva de aprendizaje (muchos parámetros), ✗ Menos interpretable |
| **Mejor para** | Competiciones, producción con precisión crítica, datos grandes con missing values |
| **Código típico** | `model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)` |

---

### 9. LightGBM (Light Gradient Boosting Machine)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `import lightgbm as lgb` → `lgb.LGBMClassifier()` |
| **Tipo** | Clasificación (Binaria y Multiclase) - Gradient Boosting ultrarrápido |
| **Cómo funciona** | Similar a XGBoost pero construye árboles hoja-por-hoja (leaf-wise) en lugar de nivel-por-nivel. Más rápido con datasets grandes. Usa histogramas para acelerar. |
| **Diferencia key** | Leaf-wise splitting vs level-wise en XGBoost |
| **Hiperparámetros clave** | `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `random_state` |
| **Mejor evaluado con** | **Primaria**: ROC-AUC, F1-Score<br>**Secundaria**: Accuracy, Precision, Recall |
| **Fortalezas** | ✓ MÁS RÁPIDO que XGBoost, ✓ Menos memoria, ✓ Excelente con datos grandes, ✓ Precisión comparable, ✓ Early stopping |
| **Debilidades** | ✗ Puede overfitear con datasets pequeños, ✗ Sensible a overfitting, ✗ Tuning requiere cuidado |
| **Mejor para** | Datasets grandes (>100k muestras), cuando velocidad es crítica, producción |
| **Código típico** | `model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, num_leaves=31, random_state=42)` |

---

### 10. CatBoost (Categorical Boosting)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `import catboost as cb` → `cb.CatBoostClassifier()` |
| **Tipo** | Clasificación (Binaria y Multiclase) - Gradient Boosting categórico-optimizado |
| **Cómo funciona** | Similar a XGBoost/LightGBM pero optimizado para variables categóricas. Maneja categorías directamente (sin encoding necesario). Antialiasing en ordenamiento. |
| **Ventaja key** | Manejo automático y óptimo de variables categóricas |
| **Hiperparámetros clave** | `n_estimators`, `max_depth`, `learning_rate`, `l2_leaf_reg`, `iterations`, `cat_features` (índices de categóricas), `random_state` |
| **Mejor evaluado con** | **Primaria**: ROC-AUC, F1-Score<br>**Secundaria**: Accuracy, Precision, Recall |
| **Fortalezas** | ✓ Mejor con categorías, ✓ Rápido, ✓ Sin necesidad de encoder, ✓ Muy preciso, ✓ Regularización automática |
| **Debilidades** | ✗ Puede ser lento en primeras iteraciones, ✗ Documentación menos extensa, ✗ Comunidad más pequeña |
| **Mejor para** | Datos con muchas categorías, cuando automatización es importante, competiciones |
| **Código típico** | `model = cb.CatBoostClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, cat_features=[0, 2, 4], random_state=42, verbose=0)` |

---

### 11. Neural Networks (Redes Neuronales - Clasificación)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.neural_network import MLPClassifier` o `import tensorflow/pytorch` |
| **Tipo** | Clasificación (Binaria y Multiclase) |
| **Cómo funciona** | Arquitectura: capas de input → capas ocultas → capa output. Cada neurona: z = w*x + b, activación = f(z). Entrena con backpropagation. Muy flexible, captura patrones complejos. |
| **Activaciones** | ReLU (ocultas), sigmoid/softmax (output), tanh |
| **Hiperparámetros clave** | `hidden_layer_sizes`, `activation`, `learning_rate`, `batch_size`, `max_iter`, `alpha` (regularización), `random_state` |
| **Mejor evaluado con** | **Primaria**: Accuracy, F1-Score, ROC-AUC<br>**Secundaria**: Precision, Recall |
| **Fortalezas** | ✓ Muy potente, ✓ Captura no-linealidad compleja, ✓ Escalable, ✓ Versátil (architecturas) |
| **Debilidades** | ✗ Caja negra (no interpretable), ✗ Requiere muchos datos, ✗ Lento en entrenamiento, ✗ Tuning complejo, ✗ Inestable |
| **Mejor para** | Datos grandes, imágenes/texto procesadas, cuando precisión máxima es crítica |
| **Código típico** | `model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)` |

---

## 📈 MODELOS DE REGRESIÓN

### 1. Linear Regression (Regresión Lineal)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.linear_model import LinearRegression` |
| **Tipo** | Regresión |
| **Cómo funciona** | Minimiza suma de residuos al cuadrado (MSE). Encuentra mejor línea ajustada: y = β₀ + β₁x₁ + ... + βₙxₙ. Cierra forma: β = (X^T X)^(-1) X^T y. Asume relación lineal. |
| **Fórmula** | ŷ = β₀ + Σ(βᵢ * xᵢ), minimiza Σ(y - ŷ)² |
| **Hiperparámetros clave** | `fit_intercept`, `normalize` (deprecated), `n_jobs` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Interpretable, ✓ Rápido, ✓ Baseline excelente, ✓ Bajo uso recursos, ✓ Coefficients interpretables |
| **Debilidades** | ✗ Asume linealidad, ✗ Sensible a outliers, ✗ Multicolinealidad problems, ✗ Mal con relaciones complejas |
| **Mejor para** | Baseline, interpretabilidad crítica, relaciones lineales conocidas |
| **Código típico** | `model = LinearRegression()` |

---

### 2. Decision Tree Regressor

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.tree import DecisionTreeRegressor` |
| **Tipo** | Regresión |
| **Cómo funciona** | Particiona feature space recursivamente. En cada nodo minimiza varianza (MSE) con split. Predice promedio de valores en hoja. Crea árboles de decisión interpretables. |
| **Criterio de split** | MSE (Mean Squared Error) o MAE |
| **Hiperparámetros clave** | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Interpretable, ✓ Maneja no-linealidad, ✓ Sin escalado, ✓ Captura interacciones, ✓ Rápido |
| **Debilidades** | ✗ Propenso a overfitting, ✗ Inestable, ✗ Bias hacia features dominantes |
| **Mejor para** | Exploración, interpretabilidad, datasets pequeños |
| **Código típico** | `model = DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42)` |

---

### 3. Random Forest Regressor

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.ensemble import RandomForestRegressor` |
| **Tipo** | Regresión - Ensemble |
| **Cómo funciona** | Múltiples árboles bootstrap con random feature subsets. Predice promediando todos los árboles. Reduce overfitting y varianza. Robusto a outliers. |
| **Agregación** | Promedio de predicciones (averaging) |
| **Hiperparámetros clave** | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `random_state`, `n_jobs` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Robusto, ✓ Maneja interacciones, ✓ Sin escalado, ✓ Feature importance, ✓ Performance consistente |
| **Debilidades** | ✗ Menos interpretable, ✗ Más lento, ✗ Más parámetros, ✗ Más memoria |
| **Mejor para** | Problemas generales, balance interpretabilidad-performance |
| **Código típico** | `model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)` |

---

### 4. Support Vector Regression (SVR)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.svm import SVR` |
| **Tipo** | Regresión |
| **Cómo funciona** | Regresión vectorial de soporte. En lugar de maximizar margen entre clases, minimiza error dentro de ε-tubo. Encuentra hiperplano que mejor ajusta datos. Usa kernel trick. |
| **Parámetro key** | ε (epsilon-tube): tolera errores dentro de este rango |
| **Hiperparámetros clave** | `C`, `epsilon`, `kernel`, `gamma`, `degree`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Efectivo en altas dimensiones, ✓ Versátil (kernels), ✓ Memoria eficiente, ✓ Good con datos pequeños |
| **Debilidades** | ✗ Lento con datasets grandes, ✗ Requiere escalado, ✗ Sensible a parámetros, ✗ Difícil tunear |
| **Mejor para** | Datasets pequeños a medianos, cuando precisión e interpretabilidad no conflictúan |
| **Código típico** | `model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')` |

---

### 5. Ridge Regression (Regresión Ridge)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.linear_model import Ridge` |
| **Tipo** | Regresión (Regularización L2) |
| **Cómo funciona** | Regresión lineal + penalidad L2 en coeficientes. Minimiza: MSE + λ * Σ(β²). Reduce overfitting penalizando coeficientes grandes. λ (alpha) controla regularización. |
| **Fórmula** | Minimizar: Σ(y - ŷ)² + α * Σ(βᵢ²) |
| **Hiperparámetros clave** | `alpha` (fuerza regularización), `fit_intercept`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Soluciona multicolinealidad, ✓ Interpretable, ✓ Rápido, ✓ Estable, ✓ Todos coefs non-zero |
| **Debilidades** | ✗ No hace feature selection, ✗ Asume linealidad |
| **Mejor para** | Datos con multicolinealidad, cuando interpretabilidad importa |
| **Código típico** | `model = Ridge(alpha=1.0, random_state=42)` |

---

### 6. Lasso Regression (Regresión Lasso)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.linear_model import Lasso` |
| **Tipo** | Regresión (Regularización L1) |
| **Cómo funciona** | Regresión lineal + penalidad L1 en coeficientes. Minimiza: MSE + λ * Σ(\|β\|). Algunos coefs → 0 (feature selection automático). λ (alpha) controla regularización. |
| **Fórmula** | Minimizar: Σ(y - ŷ)² + α * Σ(\|βᵢ\|) |
| **Hiperparámetros clave** | `alpha`, `fit_intercept`, `max_iter`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Feature selection automático, ✓ Interpretable, ✓ Algunos coefs = 0, ✓ Rápido |
| **Debilidades** | ✗ Asume linealidad, ✗ Inestable con correlacionadas, ✗ Max features limitadas (n_samples) |
| **Mejor para** | High-dimensional data, cuando necesitas feature selection, data p >> n |
| **Código típico** | `model = Lasso(alpha=0.1, random_state=42)` |

---

### 7. Elastic Net

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.linear_model import ElasticNet` |
| **Tipo** | Regresión (Regularización L1 + L2) |
| **Cómo funciona** | Combina Ridge (L2) + Lasso (L1). Minimiza: MSE + λ(ρ*L1 + (1-ρ)*L2). Mejor de ambos mundos: estabilidad + feature selection. ρ (l1_ratio) balancea L1 vs L2. |
| **Fórmula** | Minimizar: Σ(y - ŷ)² + α * (ρ*Σ(\|β\|) + (1-ρ)*Σ(β²)) |
| **Hiperparámetros clave** | `alpha`, `l1_ratio` (0=Ridge, 1=Lasso), `fit_intercept`, `max_iter`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Feature selection + estabilidad, ✓ Mejor que Lasso solo con correlacionadas, ✓ Versátil |
| **Debilidades** | ✗ Asume linealidad, ✗ Más parámetros que Ridge/Lasso |
| **Mejor para** | Correlacionadas + feature selection, cuando necesitas balance |
| **Código típico** | `model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)` |

---

### 8. Gradient Boosting Regressor

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.ensemble import GradientBoostingRegressor` |
| **Tipo** | Regresión - Gradient Boosting |
| **Cómo funciona** | Árboles secuenciales. Cada árbol predice residuos del anterior. Minimiza loss function. Actualiza predicción: F(x) = F(x) + α*f_t(x). Más preciso que RF pero más lento. |
| **Proceso** | 1. Predecir, 2. Calcular residuos, 3. Entrenar árbol en residuos, 4. Agregar α*predicción |
| **Hiperparámetros clave** | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `subsample`, `loss`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Muy preciso, ✓ Maneja interacciones, ✓ Sin escalado, ✓ Regularización |
| **Debilidades** | ✗ Lento, ✗ Overfitting risk, ✗ Más parámetros |
| **Mejor para** | Cuando precisión es crítica, datasets medianos |
| **Código típico** | `model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)` |

---

### 9. XGBoost Regressor

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `import xgboost as xgb` → `xgb.XGBRegressor()` |
| **Tipo** | Regresión - Gradient Boosting optimizado |
| **Cómo funciona** | Versión optimizada de GB. Regularización built-in. Maneja NaN. Paralelizable. Early stopping. Más rápido y preciso que GB sklearn. |
| **Objetivo** | `reg:squarederror` (default) o `reg:absoluteerror` |
| **Hiperparámetros clave** | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `lambda`, `alpha`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Muy preciso, ✓ Rápido (GPU), ✓ Regularización automática, ✓ Early stopping, ✓ NaN handling |
| **Debilidades** | ✗ Overfitting risk, ✗ Muchos parámetros, ✗ Curva aprendizaje |
| **Mejor para** | Producción, datasets grandes, precisión crítica |
| **Código típico** | `model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)` |

---

### 10. LightGBM Regressor

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `import lightgbm as lgb` → `lgb.LGBMRegressor()` |
| **Tipo** | Regresión - Gradient Boosting ultrarrápido |
| **Cómo funciona** | Similar a XGBoost pero leaf-wise splitting. Histogramas para acelerar. Muy eficiente en memoria. Excelente con datos grandes. |
| **Diferencia key** | Leaf-wise vs level-wise (XGBoost) |
| **Hiperparámetros clave** | `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ MÁS RÁPIDO, ✓ Menos memoria, ✓ Excelente grandes datos, ✓ Comparable precision |
| **Debilidades** | ✗ Overfitting con pequeños datos, ✗ Sensible tuning |
| **Mejor para** | Datasets grandes (>100k), cuando velocidad es crítica |
| **Código típico** | `model = lgb.LGBMRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, num_leaves=31, random_state=42)` |

---

### 11. CatBoost Regressor

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `import catboost as cb` → `cb.CatBoostRegressor()` |
| **Tipo** | Regresión - Gradient Boosting categórico-optimizado |
| **Cómo funciona** | Similar a XGBoost/LightGBM pero optimizado para variables categóricas. Maneja categorías directamente. Antialiasing. Muy preciso. |
| **Ventaja key** | Categóricas sin encoding necesario |
| **Hiperparámetros clave** | `n_estimators`, `max_depth`, `learning_rate`, `l2_leaf_reg`, `cat_features`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Mejor con categorías, ✓ Rápido, ✓ Sin encoder, ✓ Muy preciso |
| **Debilidades** | ✗ Comunidad más pequeña, ✗ Documentación |
| **Mejor para** | Datos con muchas categorías, automatización |
| **Código típico** | `model = cb.CatBoostRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, cat_features=[0, 2, 4], random_state=42, verbose=0)` |

---

### 12. Neural Networks (Redes Neuronales - Regresión)

| Aspecto | Detalles |
|---------|----------|
| **Librería** | `from sklearn.neural_network import MLPRegressor` o `tensorflow/pytorch` |
| **Tipo** | Regresión |
| **Cómo funciona** | Arquitectura: input → hidden layers → output (1 neurona continua). Backpropagation + optimization. Muy flexible. Captura patrones complejos. |
| **Capas output** | 1 neurona (regresión) con activación lineal o ReLU |
| **Hiperparámetros clave** | `hidden_layer_sizes`, `activation`, `learning_rate`, `batch_size`, `max_iter`, `alpha`, `random_state` |
| **Mejor evaluado con** | **Primaria**: R² Score, RMSE<br>**Secundaria**: MAE, MSE |
| **Fortalezas** | ✓ Muy potente, ✓ Captura no-linealidad, ✓ Escalable, ✓ Versátil |
| **Debilidades** | ✗ Caja negra, ✗ Requiere muchos datos, ✗ Lento, ✗ Tuning difícil, ✗ Inestable |
| **Mejor para** | Datos grandes, imágenes/audio procesados, precisión máxima |
| **Código típico** | `model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)` |

---

## 📊 TABLA RESUMEN COMPARATIVA

| Modelo | Tipo | Velocidad | Precisión | Interpretabilidad | Escalabilidad | Mejor Para |
|--------|------|-----------|-----------|-------------------|---------------|-----------|
| **Logistic Regression** | Clasificación | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Baseline, interpretabilidad |
| **Decision Tree** | Ambos | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Exploración, interpretabilidad |
| **Random Forest** | Ambos | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Propósito general, balance |
| **SVM** | Ambos | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Pequeños datos, dimensiones altas |
| **KNN** | Ambos | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | Baseline rápido, datos pequeños |
| **Naive Bayes** | Clasificación | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Texto, NLP, datos pequeños |
| **Gradient Boosting** | Ambos | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Precisión crítica, datos medianos |
| **XGBoost** | Ambos | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Competiciones, producción |
| **LightGBM** | Ambos | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Datos grandes, velocidad crítica |
| **CatBoost** | Ambos | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Muchas categorías, automatización |
| **Neural Networks** | Ambos | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | Datos enormes, complejidad máxima |
| **Ridge Regression** | Regresión | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Multicolinealidad, interpretabilidad |
| **Lasso Regression** | Regresión | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Feature selection, p >> n |
| **Elastic Net** | Regresión | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Balance Ridge + Lasso |
| **Linear Regression** | Regresión | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Baseline, interpretabilidad |
| **SVR** | Regresión | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Pequeños datos, kernels complejos |

---

## 🎯 GUÍA DE SELECCIÓN RÁPIDA

### ¿Cuál modelo elegir según tu situación?

**Si necesitas INTERPRETABILIDAD:**
→ Logistic Regression, Decision Tree, Linear Regression

**Si necesitas VELOCIDAD MÁXIMA:**
→ LightGBM, Naive Bayes, Linear/Logistic Regression

**Si necesitas MÁXIMA PRECISIÓN:**
→ XGBoost, LightGBM, CatBoost, Neural Networks

**Si tienes POCOS DATOS:**
→ SVM, KNN, Decision Tree, Naive Bayes

**Si tienes MUCHOS DATOS (>1M):**
→ LightGBM, Linear/Logistic Regression, Naive Bayes

**Si tienes MUCHAS VARIABLES CATEGÓRICAS:**
→ CatBoost, Random Forest, XGBoost

**Si datos DESBALANCEADOS:**
→ Random Forest (class_weight), XGBoost, LightGBM

**Si datos LINEALES:**
→ Linear/Logistic Regression, Ridge, Lasso

**Si datos ALTAMENTE NO-LINEALES:**
→ Random Forest, XGBoost, LightGBM, Neural Networks

**Si necesitas FEATURE SELECTION automático:**
→ Lasso, Elastic Net, Tree-based (feature importance)

---

## 💾 CÓDIGO TEMPLATE: COMPARAR MÚLTIPLES MODELOS

```python
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# CLASIFICACIÓN: Comparar múltiples modelos
clasificadores = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42),
    'CatBoost': cb.CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
}

resultados = {}
for nombre, modelo in clasificadores.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{nombre:20s} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# REGRESIÓN: Comparar múltiples modelos
regressores = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'SVR': SVR(kernel='rbf'),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
    'CatBoost': cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
}

for nombre, modelo in regressores.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{nombre:20s} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
```

---

**Última actualización**: Diciembre 2025
**Versión**: 1.0