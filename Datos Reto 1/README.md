# Reto 1 – Estimación del aporte mensual (Colsubsidio)

## Cómo reproducir
```bash
pip install -r requirements.txt
python reto1_aporte_mensual.py --data empresas_afiliadas.csv --out outputs
```
Semilla fija (42). Genera métricas, figuras, `modelo_aporte.joblib` y `resumen.json`.

## Scoring en producción
```python
from reto1_aporte_mensual import predecir
predecir(df_nuevas, "outputs/modelo_aporte.joblib")  # estimación + intervalo 80%
```

## Supuestos
1. Una fila = una empresa única; aporte en COP del mismo período. Sin nulos ni duplicados.
2. Sin dimensión temporal: modelo transversal, no es un pronóstico a futuro.
3. `salario_promedio` en millones de COP (escala 1.5–7).
4. `satisfaccion_servicio` y `numero_servicios` están disponibles al predecir (si se miden después, retirarlas).
5. Valores extremos (empresas grandes) son reales y se conservan.

## Decisiones metodológicas
- Objetivo en log (asimetría 3.3 → 0.86): el aporte es multiplicativo y el error relativo importa más que el absoluto.
- Variables: ln(trabajadores), ln(nómina total = trabajadores × salario).
- Split 70/10/20 estratificado por deciles del objetivo; test usado una sola vez.
- Comparación por CV 5-fold: baseline, Ridge, RF, XGBoost, LightGBM; se ajusta LightGBM con búsqueda aleatoria.
- Corrección de sesgo log→COP (Duan smearing) estimada en calibración.
- Intervalos 80% por conformal asimétrico (residuos con colas pesadas).
- Interpretabilidad: OLS log-log con errores robustos HC3 (explicar), permutation importance y SHAP (modelo final).

## Limitaciones
- Existe un ~2.7% de empresas que aportan ~40% más de lo que explican las variables; no son predecibles con los datos actuales (AUC 0.47). Sugiere una variable no observada.
- Cobertura real del intervalo 80% en test: 77% (colas pesadas); para presupuesto conviene usar el 90%.
- Importancias repartidas entre variables correlacionadas (trabajadores, salario, nómina).
- Sin datos temporales no se valida estabilidad en el tiempo; requiere monitoreo (drift) y reentrenamiento.
- Asociación, no causalidad.
