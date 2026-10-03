# Reto 2 – Riesgo de desafiliación de empresas (Colsubsidio)

## Cómo reproducir
```bash
pip install -r requirements.txt
python reto2_riesgo_desafiliacion.py --data base_clasificacion.csv --out outputs_reto2
```
Semilla fija (42). El CSV usa separador `;`. Genera métricas, figuras, `modelo_desafiliacion.joblib` y `resumen.json`.

## Scoring en producción
```python
from reto2_riesgo_desafiliacion import predecir
predecir(df_empresas, "outputs_reto2/modelo_desafiliacion.joblib")
# -> probabilidad, nivel de riesgo (Alto/Medio/Bajo), marca de contacto costo-beneficio y factores principales
```
`factores_principales` son las 3 variables que más empujan el riesgo hacia arriba en esa empresa (en empresas de riesgo bajo, son las que menos la protegen).

## Supuestos
1. Una fila = una empresa; `abandono`=1 es retiro en la ventana observada. Sin nulos ni duplicados.
2. Sin fechas: modelo transversal, no una curva de supervivencia.
3. `id_empresa` y `nit` se excluyen (identificadores).
4. `uso_servicios`, `satisfaccion` y `crecimiento_empleo` se miden antes del retiro. Si se miden después, el modelo sobreestima su capacidad (causalidad inversa); hay que confirmarlo.
5. Valor por empresa retenida (20 M COP/año), costo de acción (1,5 M) y efectividad (20%) son ilustrativos; reemplazar con cifras reales.
6. En el modelo explicativo, el sector de referencia es Comercio.

## Decisiones metodológicas
- Tasa de abandono 30,3%: desbalance moderado, no se re-muestrea; así las probabilidades quedan calibradas.
- Split 70/10/20 estratificado; test usado una sola vez.
- CV 5-fold sobre: baseline, logística, logística + splines, Random Forest, XGBoost, LightGBM ajustado.
- Selección por parsimonia: entre modelos a menos de 0,005 de log-loss del mejor, el más simple. Gana la regresión logística.
- Métricas: AUC, PR-AUC, log-loss, Brier, ECE (calibración), lift/ganancias por decil, IC 95% de AUC por bootstrap.
- Factores: odds ratios por +1 desv. estándar (HC3), VIF, permutation importance (caída de AUC) y SHAP.
- Decisión de negocio: contactar si p ≥ costo / (efectividad × valor), con tabla de sensibilidad.

## Limitaciones
- Asociación, no causalidad: el modelo dice quién tiene riesgo, no qué acción lo reduce. Conviene validar con un piloto con grupo de control.
- La efectividad de la acción de retención no está en los datos; es un supuesto.
- Sin datos temporales no se mide estabilidad ni el momento del retiro; requiere monitoreo (drift) y reentrenamiento.
- Sectores pequeños (Tecnología, 95 empresas en test) tienen estimaciones menos estables.
- Variables correlacionadas (antigüedad y tiempo de afiliación) comparten importancia.
