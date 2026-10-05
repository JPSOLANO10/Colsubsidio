# Reto 3 – Proyección mensual del recaudo 2026 y recomendación de presupuesto

## Cómo reproducir
```bash
pip install -r requirements.txt
python reto3_forecast_recaudo.py --data dataset_recaudo.xlsx --out outputs_reto3 --smmlv_futuro 1750905
```
Determinístico (sin aleatoriedad). `--smmlv_futuro` es el SMMLV vigente en el año a proyectar.

## Re-proyectar con datos nuevos (cada trimestre)
```python
from reto3_forecast_recaudo import proyectar
proyectar("dataset_recaudo_actualizado.xlsx", smmlv_futuro=1750905, pass_through=1.0)
```

## Modelo
recaudo = trabajadores × salario_promedio × tasa efectiva (recaudo / masa salarial).
- salario_promedio ≈ 2,6 × SMMLV (desv. 0,056 en 10 años; elasticidad 1,01, IC95% 1,00–1,02).
- trabajadores: tendencia de los últimos 24 meses (en logaritmos).
- tasa efectiva: Holt-Winters amortiguado con estacionalidad anual.

## Supuestos
1. Serie mensual completa 2015-01 a 2025-12, en COP corrientes; se proyecta 2026.
2. El SMMLV 2026 se conoce al presupuestar: 1.750.905 (+23,0%, Decreto 1469 de 2025). Es parámetro.
3. La relación salario_promedio / SMMLV se mantiene en el nivel de los últimos 36 meses.
4. Escenarios conservador y estrés son ilustrativos (pass-through 70%; choques de empleo y tasa calibrados con 2020, a la mitad y completos). Validar con finanzas.
5. Las variables macro se usan para explicar, no se proyectan.

## Validación
Backtest rodante: cada diciembre 2018–2024 se proyectan los 12 meses siguientes (7 folds). Sin la ventana de choque 2020–21 (5 folds), el MAPE del modelo estructural es 3,1% frente a 3,7% de la referencia estacional con deriva, 4,6% de SARIMA y 5,3% de Holt-Winters. Con los 7 folds sube a 6,8%, porque ningún modelo anticipó el choque de 2020. Si se desconoce el SMMLV, el estructural pasa a 10,2%.

## Limitaciones
- El reajuste de 2026 (+23%) supera el máximo de la muestra (+16%): la transmisión al salario promedio es una extrapolación.
- Con 5 folds sin choque, la diferencia frente a la referencia estacional con deriva no es concluyente en el error anual total (2,3% vs 1,7%); la ventaja del estructural es que reacciona al SMMLV.
- Las bandas (P10–P90) vienen de 60 errores mensuales de 5 años, correlacionados entre sí.
- Un choque tipo 2020 solo está representado por un escenario, no por la banda.
- La relación con PIB y desempleo es de asociación y opera vía empleo; no se proyectan.
- Precios corrientes: no se separa efecto inflacionario del real.

## Seguimiento recomendado
Revisar mensualmente: (i) k = salario_promedio / SMMLV (debería ser ~2,6; si cae por debajo de ~2,45 durante dos meses, pasar al escenario conservador); (ii) desviación del recaudo vs. Base (alerta si supera −5,6% o +7,7%, los P5/P95 del backtest).
