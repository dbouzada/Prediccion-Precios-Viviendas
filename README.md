# 🏠 Predicción de Precios de Viviendas — Ames, Iowa

Análisis y modelado predictivo sobre el precio de viviendas residenciales
en Ames, Iowa (dataset Kaggle House Prices).
Proyecto de cierre del curso de Data Science en Coderhouse.

---

## 🎯 Objetivo

Predecir el precio de venta (`SalePrice`) de viviendas residenciales
a partir de variables estructurales y de calidad, usando técnicas de
regresión supervisada con optimización de hiperparámetros.

---

## 📦 Dataset

- **Fuente:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Registros originales:** 1.460 viviendas · 81 variables
- **Registros tras limpieza:** 1.331 viviendas · 10 variables seleccionadas
- **Variable objetivo:** `SalePrice` (precio de venta en USD)

---

## 🔍 Proceso de Análisis

### 1. Exploración y selección de variables
Se generó un mapa de correlación completo (81 variables) para identificar
las features con correlación > 0.5 respecto a `SalePrice`:

| Variable | Descripción |
|---|---|
| `OverallQual` | Calidad general de materiales y terminaciones |
| `GrLivArea` | Superficie habitable sobre el nivel del suelo (ft²) |
| `GarageArea` | Superficie del garage (ft²) |
| `GarageCars` | Capacidad del garage (autos) |
| `TotalBsmtSF` | Superficie total del sótano (ft²) |
| `1stFlrSF` | Superficie del primer piso (ft²) |
| `YearBuilt` | Año de construcción |
| `YearRemodAdd` | Año de última remodelación |
| `TotRmsAbvGrd` | Total de habitaciones sobre el nivel del suelo |

### 2. Limpieza y tratamiento de outliers
Se eliminaron registros con valores extremos en:
- `SalePrice` > USD 400.000 (28 registros)
- `GrLivArea` > 3.000 ft²
- `GarageArea` > 1.000 ft² o igual a 0
- `TotalBsmtSF` > 2.300 ft²

### 3. Modelado
Se entrenaron tres modelos de regresión con optimización de
hiperparámetros via **GridSearchCV**:

---

## 📊 Resultados por Modelo

| Modelo | R² (Test) | Notas |
|---|---|---|
| Decision Tree Regressor | ~50% | Sobreajuste severo en train (100%) |
| Linear Regression | **83.52%** | Buen balance entre sesgo y varianza |
| Gradient Boosting Regressor | **87.44%** | Mejor modelo — parámetros optimizados |

### Mejores parámetros — Gradient Boosting (GridSearchCV):
```python
