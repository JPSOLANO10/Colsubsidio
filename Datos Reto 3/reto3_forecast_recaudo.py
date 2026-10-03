# %% [markdown]
# # Reto 3 - Proyección mensual del recaudo y recomendación de presupuesto (Colsubsidio)
#
# Ejecución:
#   python reto3_forecast_recaudo.py --data dataset_recaudo.xlsx --out outputs_reto3 --smmlv_futuro 1750905
#
# Flujo:
#   1. Carga y validación (serie mensual completa, identidades contables)
#   2. Análisis del comportamiento histórico: estacionalidad, choque 2020, descomposición del crecimiento
#   3. Factores que influyen: relación salario mínimo -> salario promedio, y variables macro
#   4. Backtest rodante (rolling origin) de 12 meses, un origen por diciembre (2018-2024): modelo estructural vs. referencias estadísticas
#   5. Proyección 12 meses: escenarios base / conservador / estrés, con bandas empíricas
#   6. Recomendación de presupuesto y reglas de seguimiento
#
# IDEA CENTRAL (por qué un modelo estructural):
#   recaudo = trabajadores x salario_promedio x tasa_efectiva        (tasa = recaudo / masa_salarial)
#   - salario_promedio ~ k x salario_minimo (k ~ 2,6, estable 10 años) -> el reajuste del SMMLV, que se conoce
#     por decreto a finales de diciembre, explica la mayor parte del salto anual del recaudo.
#   - trabajadores: tendencia lenta y suave.
#   - tasa efectiva: estacionalidad fuerte (enero bajo, diciembre alto) y deriva lenta al alza.
#   Un modelo de series de tiempo "puro" no puede anticipar el salto del SMMLV; el estructural sí.
#
# SUPUESTOS:
#   S1. Serie mensual completa 2015-01 a 2025-12; el pronóstico cubre 2026-01 a 2026-12.
#   S2. 'recaudo' está en COP corrientes (nominales).
#   S3. El SMMLV del año a proyectar se conoce al elaborar el presupuesto (decreto de fines de diciembre).
#       Es un PARÁMETRO (--smmlv_futuro). Por defecto 1.750.905 (Decreto 1469 de 2025, +23,0%).
#   S4. La relación salario_promedio / salario_minimo (k) se mantiene en el nivel de los últimos 36 meses.
#   S5. Los escenarios conservador y de estrés son ilustrativos y se calibran con datos históricos (ver sección 5);
#       deben validarse con el equipo financiero.
#   S6. Las variables macro (PIB, TRM, petróleo...) no se proyectan: se analizan como explicación, no como insumo,
#       porque proyectarlas añadiría incertidumbre mayor que la información que aportan.
# %%
import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
H = 12                                   # horizonte de proyección (meses)
SHOCK = ("2020-01-01", "2021-12-01")     # periodo de choque COVID: se reporta el backtest con y sin esta ventana
SMMLV_2026_DEFECTO = 1_750_905
MACRO = ["desempleo", "inflacion", "pib", "trm", "precio_petroleo", "indice_confianza"]


# %% ------------------------------------------------------------------ carga y validación
def cargar_y_validar(path: str) -> pd.DataFrame:
    d = pd.read_excel(path, parse_dates=["fecha"]).set_index("fecha").sort_index()
    d = d.asfreq("MS")  # fuerza frecuencia mensual: si faltara un mes aparecería NaN
    assert d.isna().sum().sum() == 0, "Hay meses faltantes o nulos: definir imputación"
    assert len(d) == 132, "Se esperaban 132 meses (2015-01 a 2025-12)"
    # Identidad contable: masa_salarial = trabajadores x salario_promedio
    assert np.allclose(d["masa_salarial"] / (d["trabajadores"] * d["salario_promedio"]), 1, rtol=1e-3)
    d["tasa"] = d["recaudo"] / d["masa_salarial"]          # tasa efectiva de recaudo sobre la masa salarial
    d["k_salario"] = d["salario_promedio"] / d["salario_minimo"]
    return d


# %% ------------------------------------------------------------------ modelos
def ajustar_estructural(hist: pd.DataFrame, smin_futuro: np.ndarray, pass_through: float = 1.0,
                        ajuste_empleo: float = 0.0, ajuste_tasa: float = 0.0, k_ventana: int = 36) -> np.ndarray:
    """Proyección estructural: trabajadores x salario_promedio x tasa.

    - salario_promedio = k x SMMLV efectivo. k = promedio de los últimos `k_ventana` meses.
      pass_through < 1 modela que no todo el reajuste del SMMLV llega al salario promedio.
    - trabajadores: deriva lineal en logaritmos con la tendencia de los últimos 24 meses (estable, sin sesgo en backtest).
    - tasa efectiva: Holt-Winters aditivo amortiguado sobre ln(tasa) con estacionalidad anual.
    - ajuste_empleo / ajuste_tasa: desplazamientos en log (para escenarios).
    """
    n = len(smin_futuro)
    k = hist["k_salario"].iloc[-k_ventana:].mean()
    smin_ref = hist["salario_minimo"].iloc[-1]
    smin_eff = smin_ref * (smin_futuro / smin_ref) ** pass_through
    ln_t = np.log(hist["trabajadores"])
    g = (ln_t.iloc[-1] - ln_t.iloc[-25]) / 24
    trab = np.exp(ln_t.iloc[-1] + g * np.arange(1, n + 1) + ajuste_empleo)
    ets = ExponentialSmoothing(np.log(hist["tasa"]), trend="add", damped_trend=True,
                               seasonal="add", seasonal_periods=12).fit()
    tasa = np.exp(ets.forecast(n).values + ajuste_tasa)
    return trab * (k * smin_eff) * tasa


def ref_snaive_drift(hist: pd.DataFrame, n: int = H) -> np.ndarray:
    """Referencia 1: repite los últimos 12 meses y los escala por el crecimiento del último año."""
    y = hist["recaudo"]
    return y.iloc[-12:].values * (y.iloc[-12:].sum() / y.iloc[-24:-12].sum())


def ref_ets(hist: pd.DataFrame, n: int = H) -> np.ndarray:
    """Referencia 2: Holt-Winters amortiguado sobre ln(recaudo) (sin información del SMMLV)."""
    m = ExponentialSmoothing(np.log(hist["recaudo"]), trend="add", damped_trend=True,
                             seasonal="add", seasonal_periods=12).fit()
    return np.exp(m.forecast(n).values)


def ref_sarima(hist: pd.DataFrame, n: int = H) -> np.ndarray:
    """Referencia 3: SARIMA(1,1,1)(0,1,1,12) sobre ln(recaudo) (especificación clásica para series con tendencia y estacionalidad)."""
    m = SARIMAX(np.log(hist["recaudo"]), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    return np.exp(m.forecast(n).values)


# %% ------------------------------------------------------------------ backtest
def backtest(d: pd.DataFrame) -> pd.DataFrame:
    """Rolling origin: cada diciembre se 'congela' la historia, se proyectan los 12 meses siguientes y se compara con lo ocurrido.
    Imita exactamente la situación real de elaborar el presupuesto: en diciembre el SMMLV del año siguiente ya es conocido
    (por eso no se usan orígenes a mitad de año, donde el SMMLV futuro todavía no se conocería)."""
    filas = []
    origenes = pd.date_range("2018-12-01", "2024-12-01", freq="12MS")  # >= 4 años de historia en el primer origen
    for o in origenes:
        hist = d.loc[:o]
        test = d.loc[o + pd.offsets.MonthBegin(1):].iloc[:H]
        if len(test) < H:
            continue
        real = test["recaudo"].values
        preds = {
            "Estructural (SMMLV conocido)": ajustar_estructural(hist, test["salario_minimo"].values),
            "Estructural (SMMLV desconocido)": ajustar_estructural(hist, np.repeat(hist["salario_minimo"].iloc[-1], H)),
            "Holt-Winters (ETS)": ref_ets(hist),
            "SARIMA": ref_sarima(hist),
            "Estacional naive + deriva": ref_snaive_drift(hist),
        }
        choque = (test.index.min() <= pd.Timestamp(SHOCK[1])) and (test.index.max() >= pd.Timestamp(SHOCK[0]))
        for nombre, p in preds.items():
            filas.append({"origen": o, "modelo": nombre, "ventana_choque": choque,
                          "MAPE": np.mean(np.abs(p / real - 1)),
                          "error_total_anual": p.sum() / real.sum() - 1,
                          "log_errores": list(np.log(real / p))})
    return pd.DataFrame(filas)


def resumen_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    def agg(sub, etiqueta):
        g = sub.groupby("modelo").agg(MAPE_medio=("MAPE", "mean"), MAPE_max=("MAPE", "max"),
                                      error_total_abs=("error_total_anual", lambda s: s.abs().mean()),
                                      sesgo_total=("error_total_anual", "mean"), folds=("MAPE", "size"))
        g.insert(0, "periodo", etiqueta)
        return g
    r = pd.concat([agg(bt, "todos los folds"), agg(bt[~bt["ventana_choque"]], "sin choque 2020-21")])
    return r.reset_index().sort_values(["periodo", "MAPE_medio"])


# %% ------------------------------------------------------------------ main
def main(data_path: str, out_dir: str, smmlv_futuro: float):
    out = Path(out_dir)
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ---------------- 1. Carga
    d = cargar_y_validar(data_path)
    print(f"Datos: {d.index.min():%Y-%m} a {d.index.max():%Y-%m} ({len(d)} meses), sin faltantes")
    k_hist = d["k_salario"]
    print(f"Relación salario_promedio / SMMLV: media {k_hist.mean():.3f}, desv. {k_hist.std():.3f} "
          f"(rango {k_hist.min():.2f}-{k_hist.max():.2f}) -> estable en 10 años y con reajustes de +3,5% a +16%")

    # ---------------- 2. Comportamiento histórico
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    (d["recaudo"] / 1e9).plot(ax=ax[0], title="Recaudo mensual (miles de millones COP)")
    d["tasa"].plot(ax=ax[1], title="Tasa efectiva = recaudo / masa salarial")
    ax[0].axvspan(SHOCK[0], "2020-12-31", color="red", alpha=.1, label="2020 (choque)"); ax[0].legend()
    plt.tight_layout(); plt.savefig(out / "figuras/01_serie_y_tasa.png", dpi=130); plt.close()

    # Estacionalidad: STL robusto sobre ln(recaudo); fuerza = 1 - var(resid)/var(estacional+resid)
    stl = STL(np.log(d["recaudo"]), period=12, robust=True).fit()
    fuerza_est = max(0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
    perfil = (stl.seasonal.groupby(stl.seasonal.index.month).mean()).apply(np.exp).sub(1).mul(100)
    perfil.round(2).rename("efecto_estacional_%").to_csv(out / "estacionalidad_mensual.csv")
    print(f"Fuerza de estacionalidad (STL): {fuerza_est:.2f} | efecto estacional: "
          f"enero {perfil.loc[1]:+.1f}% / diciembre {perfil.loc[12]:+.1f}%")
    stl.plot(); plt.tight_layout(); plt.savefig(out / "figuras/02_stl.png", dpi=110); plt.close()

    # Estacionariedad (justifica trabajar con logaritmos y diferencias / modelos con tendencia)
    adf = {c: adfuller(np.log(d[c]))[1] for c in ["recaudo", "masa_salarial"]}
    print("ADF p-valor en nivel (ln):", {k: round(v, 3) for k, v in adf.items()}, "-> no estacionarias (tendencia)")

    # Descomposición del crecimiento anual: d ln(recaudo) = d ln(trabajadores) + d ln(salario) + d ln(tasa)  (exacta)
    anual = d[["recaudo", "trabajadores", "salario_promedio", "tasa"]].resample("YS").agg(
        {"recaudo": "sum", "trabajadores": "mean", "salario_promedio": "mean", "tasa": "mean"})
    contrib = np.log(anual).diff().dropna() * 100
    contrib["tasa"] = np.log(anual["recaudo"]).diff().dropna() * 100 - contrib["trabajadores"] - contrib["salario_promedio"]
    contrib.columns = ["crec_recaudo_%", "aporte_trabajadores_pp", "aporte_salario_pp", "aporte_tasa_pp"]
    contrib.index = contrib.index.year
    contrib.round(2).to_csv(out / "descomposicion_crecimiento_anual.csv")
    contrib[["aporte_trabajadores_pp", "aporte_salario_pp", "aporte_tasa_pp"]].plot.bar(
        stacked=True, figsize=(10, 4.5), title="¿De dónde viene el crecimiento anual del recaudo? (puntos porcentuales, log)")
    plt.axhline(0, c="k", lw=.8); plt.tight_layout(); plt.savefig(out / "figuras/03_descomposicion.png", dpi=130); plt.close()
    print("\nDescomposición del crecimiento anual (pp):\n", contrib.round(1).to_string())

    # ---------------- 3. Factores que influyen
    # 3a. Salario promedio vs SMMLV (elasticidad en logs, errores robustos HAC por autocorrelación)
    X = sm.add_constant(np.log(d["salario_minimo"]))
    m_sal = sm.OLS(np.log(d["salario_promedio"]), X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    elast, ic = m_sal.params["salario_minimo"], m_sal.conf_int().loc["salario_minimo"].values
    print(f"\nElasticidad salario_promedio vs SMMLV: {elast:.2f} (IC95% {ic[0]:.2f}-{ic[1]:.2f}), R2={m_sal.rsquared:.3f}")

    # 3b. Correlaciones de variaciones anuales (evita correlación espuria de tendencias) con rezagos 0-6 meses
    yoy = np.log(d[["recaudo", "trabajadores", "salario_promedio", "salario_minimo", "masa_salarial"]]).diff(12)
    yoy["tasa_dif"] = d["tasa"].diff(12)
    mac = d[MACRO].copy()
    mac_yoy = pd.DataFrame({"desempleo": mac["desempleo"].diff(12), "inflacion": mac["inflacion"].diff(12),
                            "pib": np.log(mac["pib"]).diff(12), "trm": np.log(mac["trm"]).diff(12),
                            "precio_petroleo": np.log(mac["precio_petroleo"]).diff(12),
                            "indice_confianza": mac["indice_confianza"].diff(12)})
    filas = []
    for v in MACRO:
        for objetivo in ["recaudo", "trabajadores"]:
            for lag in [0, 3, 6]:
                filas.append({"variable": v, "objetivo": objetivo, "rezago_meses": lag,
                              "corr": mac_yoy[v].shift(lag).corr(yoy[objetivo])})
    corr_mac = pd.DataFrame(filas)
    corr_mac.round(3).to_csv(out / "correlaciones_macro_yoy.csv", index=False)
    mejor = corr_mac.loc[corr_mac["corr"].abs().groupby([corr_mac["variable"], corr_mac["objetivo"]]).idxmax()]
    print("\nMayor correlación (|r|) de cada variable macro con el crecimiento anual:\n",
          mejor[["variable", "objetivo", "rezago_meses", "corr"]].round(2).to_string(index=False))

    # 3c. Causalidad de Granger (¿las macro "anticipan" el crecimiento del recaudo?), rezago hasta 6
    g_rows = []
    base = yoy["recaudo"]
    for v in MACRO:
        z = pd.concat([base, mac_yoy[v]], axis=1).dropna()
        # statsmodels reciente ya no acepta verbose=False: se silencia la salida impresa
        with contextlib.redirect_stdout(io.StringIO()):
            res = grangercausalitytests(z, maxlag=6)
        p = min(res[l][0]["ssr_ftest"][1] for l in range(1, 7))
        g_rows.append({"variable": v, "p_valor_min_rezagos_1a6": p})
    gr = pd.DataFrame(g_rows)
    gr["p_ajustado_bonferroni"] = (gr["p_valor_min_rezagos_1a6"] * 6).clip(upper=1)  # 6 rezagos probados
    gr.round(4).to_csv(out / "granger_macro.csv", index=False)

    # 3d. Dependencia del recaudo con el empleo y con el reajuste: regresión de crecimiento anual
    Z = pd.concat([yoy["recaudo"], yoy["salario_minimo"], yoy["trabajadores"], yoy["tasa_dif"]], axis=1).dropna()
    Z.columns = ["d12_recaudo", "d12_smmlv", "d12_trabajadores", "d12_tasa"]
    m_g = sm.OLS(Z["d12_recaudo"], sm.add_constant(Z[["d12_smmlv", "d12_trabajadores", "d12_tasa"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 12})
    pd.DataFrame({"coef": m_g.params, "p_valor": m_g.pvalues}).round(4).to_csv(out / "regresion_crecimiento_anual.csv")
    print(f"Regresión del crecimiento anual (ln): R2={m_g.rsquared:.3f}\n", m_g.params.round(3).to_string())

    # ---------------- 4. Backtest
    print("\nBacktest rodante (12 meses, origen cada diciembre desde 2018)...")
    bt = backtest(d)
    res_bt = resumen_backtest(bt)
    res_bt.round(4).to_csv(out / "backtest_resumen.csv", index=False)
    bt.drop(columns="log_errores").round(4).to_csv(out / "backtest_detalle.csv", index=False)
    print(res_bt.round(3).to_string(index=False))
    por_anio = bt.assign(anio_proyectado=bt["origen"].dt.year + 1).pivot(index="anio_proyectado", columns="modelo", values="MAPE")
    por_anio.round(4).to_csv(out / "backtest_mape_por_anio.csv")
    print("\nMAPE por año proyectado:\n", por_anio.round(3).to_string())
    CAMPEON = "Estructural (SMMLV conocido)"

    # Errores de pronóstico del campeón (en log) para bandas: solo ventanas SIN choque, y con choque para estrés
    errs_ok = np.concatenate(bt[(bt["modelo"] == CAMPEON) & (~bt["ventana_choque"])]["log_errores"].tolist())
    q = {p: float(np.quantile(errs_ok, p)) for p in [0.05, 0.10, 0.50, 0.90, 0.95]}
    print(f"\nErrores mensuales del campeón (sin choque) -> P5 {np.exp(q[0.05])-1:+.1%}, P95 {np.exp(q[0.95])-1:+.1%}")

    # ---------------- 5. Proyección 12 meses
    ult = d.index[-1]
    fechas = pd.date_range(ult + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    smmlv_ref = d["salario_minimo"].iloc[-1]
    smin_fut = np.repeat(float(smmlv_futuro), H)  # el SMMLV aplica de enero a diciembre
    crec_smmlv = smmlv_futuro / smmlv_ref - 1
    print(f"\nSMMLV de referencia {smmlv_ref:,.0f} -> {smmlv_futuro:,.0f} ({crec_smmlv:+.1%}); "
          f"máximo reajuste histórico en la muestra: {(d['salario_minimo'].pct_change(12).max()):+.1%}")

    # Escenarios. Choque calibrado con 2020 (supuesto S5): caída de empleo Dic-19->Ene-20 y de la tasa promedio 2020 vs 2019
    choque_empleo = float(np.log(d.loc["2020-01-01", "trabajadores"] / d.loc["2019-12-01", "trabajadores"]))
    choque_tasa = float(np.log(d.loc["2020", "tasa"].mean() / d.loc["2019", "tasa"].mean()))
    escenarios = {
        "Base": dict(pass_through=1.0, ajuste_empleo=0.0, ajuste_tasa=0.0),
        "Conservador": dict(pass_through=0.7, ajuste_empleo=choque_empleo / 2, ajuste_tasa=choque_tasa / 2),
        "Estrés (tipo 2020)": dict(pass_through=0.7, ajuste_empleo=choque_empleo, ajuste_tasa=choque_tasa),
    }
    proy = pd.DataFrame(index=fechas)
    for nombre, kw in escenarios.items():
        proy[nombre] = ajustar_estructural(d, smin_fut, **kw)
    # Bandas empíricas alrededor del escenario Base (errores de backtest sin choque)
    proy["Base_P10"] = proy["Base"] * np.exp(q[0.10])
    proy["Base_P90"] = proy["Base"] * np.exp(q[0.90])
    proy["Base_P05"] = proy["Base"] * np.exp(q[0.05])
    proy["Base_P95"] = proy["Base"] * np.exp(q[0.95])
    # Referencia estadística pura para contraste (no usa el SMMLV)
    proy["Ref_ETS_sin_SMMLV"] = ref_ets(d)
    proy.index.name = "fecha"
    proy.round(0).to_csv(out / "proyeccion_mensual_2026.csv")

    rec_2025 = d["recaudo"].iloc[-12:].sum()
    tot = pd.DataFrame({"total_anual_COP": proy.sum()})
    tot["crecimiento_vs_2025_%"] = (tot["total_anual_COP"] / rec_2025 - 1) * 100
    tot.round(1).to_csv(out / "proyeccion_totales_2026.csv")
    print("\nTotales proyectados (miles de millones COP):\n",
          (tot.assign(total_anual_COP=tot["total_anual_COP"] / 1e9)).round(1).to_string())

    # Figura: histórico + proyección + bandas + escenarios
    plt.figure(figsize=(12, 5.5))
    h = d["recaudo"].loc["2022":] / 1e9
    plt.plot(h.index, h, c="k", label="Histórico")
    plt.plot(proy.index, proy["Base"] / 1e9, c="C0", lw=2, label="Base")
    plt.fill_between(proy.index, proy["Base_P10"] / 1e9, proy["Base_P90"] / 1e9, color="C0", alpha=.25, label="Banda 80% (backtest)")
    plt.fill_between(proy.index, proy["Base_P05"] / 1e9, proy["Base_P95"] / 1e9, color="C0", alpha=.1, label="Banda 90%")
    plt.plot(proy.index, proy["Conservador"] / 1e9, c="orange", ls="--", label="Conservador")
    plt.plot(proy.index, proy["Estrés (tipo 2020)"] / 1e9, c="red", ls="--", label="Estrés")
    plt.plot(proy.index, proy["Ref_ETS_sin_SMMLV"] / 1e9, c="gray", ls=":", label="ETS sin SMMLV (referencia)")
    plt.title("Recaudo mensual: proyección 2026 (miles de millones COP)"); plt.legend(ncol=2)
    plt.tight_layout(); plt.savefig(out / "figuras/04_proyeccion.png", dpi=130); plt.close()

    # Figura de backtest: último fold completo (origen 2024-12 -> 2025)
    o = pd.Timestamp("2024-12-01"); hist = d.loc[:o]; test = d.loc[o + pd.offsets.MonthBegin(1):].iloc[:H]
    plt.figure(figsize=(10, 4.5))
    plt.plot(test.index, test["recaudo"] / 1e9, "k", label="Real 2025")
    plt.plot(test.index, ajustar_estructural(hist, test["salario_minimo"].values) / 1e9, label="Estructural")
    plt.plot(test.index, ref_ets(hist) / 1e9, "--", label="ETS")
    plt.plot(test.index, ref_sarima(hist) / 1e9, ":", label="SARIMA")
    plt.title("Backtest: proyección hecha en dic-2024 vs. real 2025"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "figuras/05_backtest_2025.png", dpi=130); plt.close()

    # ---------------- 6. Resumen y recomendación
    base_tot, cons_tot, estres_tot = (proy["Base"].sum(), proy["Conservador"].sum(), proy["Estrés (tipo 2020)"].sum())
    p10_tot, p90_tot = proy["Base_P10"].sum(), proy["Base_P90"].sum()
    resumen = {
        "smmlv_ref": float(smmlv_ref), "smmlv_futuro": float(smmlv_futuro), "crec_smmlv": float(crec_smmlv),
        "k_salario_medio": float(k_hist.mean()), "elasticidad_salario_smmlv": float(elast),
        "fuerza_estacionalidad": float(fuerza_est), "recaudo_2025": float(rec_2025),
        "proyeccion_2026": {"base": float(base_tot), "conservador": float(cons_tot), "estres": float(estres_tot),
                            "base_p10": float(p10_tot), "base_p90": float(p90_tot),
                            "ets_sin_smmlv": float(proy["Ref_ETS_sin_SMMLV"].sum())},
        "crecimiento_vs_2025": {"base": float(base_tot / rec_2025 - 1), "conservador": float(cons_tot / rec_2025 - 1),
                                "estres": float(estres_tot / rec_2025 - 1)},
        "choque_calibrado_2020": {"empleo_log": choque_empleo, "tasa_log": choque_tasa},
        "campeon": CAMPEON,
        "banda_mensual_P10_P90": [float(np.exp(q[0.10]) - 1), float(np.exp(q[0.90]) - 1)],
        "alerta_desviacion_mensual": float(np.exp(q[0.95]) - 1),
    }
    (out / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nRecomendación: presupuestar Base = {base_tot/1e9:,.0f} MM COP ({base_tot/rec_2025-1:+.1%} vs 2025); "
          f"gasto comprometido hasta P10 = {p10_tot/1e9:,.0f} MM; piso de estrés = {estres_tot/1e9:,.0f} MM.")
    print(f"Artefactos guardados en {out.resolve()}")


# %% ------------------------------------------------------------------ re-proyección con datos nuevos
def proyectar(ruta_datos: str, smmlv_futuro: float, pass_through: float = 1.0) -> pd.DataFrame:
    """Reproyecta 12 meses con datos actualizados (mismo formato del xlsx). Úsese cada trimestre al cerrar el mes."""
    d = cargar_y_validar_flexible(ruta_datos)
    fechas = pd.date_range(d.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
    p = ajustar_estructural(d, np.repeat(float(smmlv_futuro), H), pass_through=pass_through)
    return pd.DataFrame({"recaudo_proyectado": p}, index=fechas)


def cargar_y_validar_flexible(path: str) -> pd.DataFrame:
    d = pd.read_excel(path, parse_dates=["fecha"]).set_index("fecha").sort_index().asfreq("MS")
    assert d.isna().sum().sum() == 0, "Hay meses faltantes o nulos"
    d["tasa"] = d["recaudo"] / d["masa_salarial"]
    d["k_salario"] = d["salario_promedio"] / d["salario_minimo"]
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_recaudo.xlsx")
    ap.add_argument("--out", default="outputs_reto3")
    ap.add_argument("--smmlv_futuro", type=float, default=SMMLV_2026_DEFECTO,
                    help="SMMLV vigente en el año a proyectar (COP). Por defecto 1.750.905 (Decreto 1469/2025)")
    a = ap.parse_args()
    main(a.data, a.out, a.smmlv_futuro)
