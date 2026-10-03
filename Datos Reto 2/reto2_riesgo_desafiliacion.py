# %% [markdown]
# # Reto 2 - Predicción del riesgo de desafiliación de empresas (Colsubsidio)
#
# Ejecución:
#   python reto2_riesgo_desafiliacion.py --data base_clasificacion.csv --out outputs_reto2
#
# Flujo:
#   1. Carga y validación (separador ';', sin fugas ni identificadores)
#   2. EDA: tasa de abandono por rangos de cada variable (lectura de negocio)
#   3. Ingeniería de variables
#   4. Partición estratificada train / calibración / test
#   5. Comparación de modelos por CV (AUC, PR-AUC, log-loss, Brier) y regla de parsimonia
#   6. Evaluación final, calibración, intervalos bootstrap, error por segmento
#   7. Valor de negocio: lift/ganancias por decil y decisión de a cuántas empresas contactar
#   8. Factores asociados: odds ratios (logit), permutation importance y SHAP
#   9. Persistencia + función de scoring con "motivos" del riesgo
#
# SUPUESTOS:
#   S1. Cada fila es una empresa única; 'abandono'=1 significa que se desafilió en la ventana observada.
#   S2. No hay fechas: el modelo es transversal (estado actual -> abandono observado). No es una curva de supervivencia.
#   S3. 'id_empresa' y 'nit' son identificadores: se excluyen (no generalizan; el NIT podría filtrar información).
#   S4. 'uso_servicios', 'satisfaccion' y 'crecimiento_empleo' se miden ANTES del retiro. Si se midieran
#       después (p. ej. una empresa que se va deja de usar servicios), habría causalidad inversa y el modelo
#       sobreestimaría su capacidad. Debe confirmarse con el área de datos.
#   S5. Los parámetros económicos de la sección 7 (valor, costo, efectividad) son ILUSTRATIVOS y deben
#       reemplazarse por cifras reales del negocio.
# %%
import argparse
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "abandono"
CAT = ["sector"]
IDS = ["id_empresa", "nit"]

# Parámetros económicos ILUSTRATIVOS (COP) -> reemplazar con datos reales (supuesto S5)
VALOR_EMPRESA_RETENIDA = 20_000_000   # valor anual esperado de conservar una empresa
COSTO_ACCION = 1_500_000              # costo de una acción de retención por empresa
TOL_LOGLOSS = 0.005                   # diferencia en log-loss considerada sin relevancia práctica
EFECTIVIDAD = 0.20                    # prob. de que la acción evite el retiro de una empresa que se iría


# %% ------------------------------------------------------------------ utilidades
def cargar_y_validar(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    assert df["id_empresa"].is_unique and df["nit"].is_unique, "Identificadores duplicados"
    assert df.isna().sum().sum() == 0, "Hay nulos: definir imputación"
    assert set(df[TARGET].unique()) <= {0, 1}, "Objetivo no binario"
    assert (df["tiempo_afiliacion"] <= df["antiguedad"]).all(), "Inconsistencia: afiliación > antigüedad"
    return df


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Misma función en entrenamiento y producción.
    - ln_trabajadores: tamaño muy asimétrico (5 a 500) -> log.
    - ratio_afiliacion: fracción de la vida de la empresa que lleva afiliada (afiliación tardía vs. temprana).
    """
    out = df.drop(columns=[c for c in IDS + [TARGET] if c in df.columns]).copy()
    out["ln_trabajadores"] = np.log(out["num_trabajadores"])
    out["ratio_afiliacion"] = out["tiempo_afiliacion"] / out["antiguedad"]
    out["sector"] = out["sector"].astype("category")
    return out


def ece(y, p, bins=10) -> float:
    """Error de calibración esperado: qué tan cerca está la probabilidad predicha de la frecuencia real."""
    frac, medio = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return float(np.mean(np.abs(frac - medio)))


def metricas(y, p) -> dict:
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "LogLoss": log_loss(y, p), "Brier": brier_score_loss(y, p), "ECE": ece(y, p)}


def tabla_ganancias(y, p, deciles=10) -> pd.DataFrame:
    """Decil 1 = empresas de mayor riesgo. Lift = tasa del decil / tasa global."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decil"] = (np.arange(len(d)) * deciles // len(d)) + 1
    g = d.groupby("decil").agg(empresas=("y", "size"), abandonan=("y", "sum"),
                               tasa_real=("y", "mean"), prob_media=("p", "mean"))
    g["lift"] = g["tasa_real"] / d["y"].mean()
    g["captura_acum_%"] = 100 * g["abandonan"].cumsum() / d["y"].sum()
    return g


# %% ------------------------------------------------------------------ main
def main(data_path: str, out_dir: str):
    out = Path(out_dir)
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ---------------- 1. Carga
    raw = cargar_y_validar(data_path)
    tasa = raw[TARGET].mean()
    print(f"Datos: {len(raw):,} empresas | tasa de abandono {tasa:.1%} (desbalance moderado: no se re-muestrea)")

    # ---------------- 2. EDA: tasa de abandono por quintiles (lectura directa para el negocio)
    num_raw = [c for c in raw.columns if c not in IDS + [TARGET] + CAT]
    filas = []
    for c in num_raw:
        q = pd.qcut(raw[c], 5, duplicates="drop")
        t = raw.groupby(q, observed=True)[TARGET].mean()
        filas.append({"variable": c, "tasa_Q1_(bajo)": t.iloc[0], "tasa_Q5_(alto)": t.iloc[-1],
                      "diferencia_pp": 100 * (t.iloc[-1] - t.iloc[0])})
    eda = pd.DataFrame(filas).set_index("variable").sort_values("diferencia_pp")
    eda.round(3).to_csv(out / "eda_tasa_abandono_quintiles.csv")
    raw.groupby("sector")[TARGET].agg(["mean", "count"]).round(3).to_csv(out / "eda_tasa_por_sector.csv")
    eda["diferencia_pp"].plot.barh(figsize=(7, 5), title="Cambio en tasa de abandono: quintil alto vs. bajo (pp)")
    plt.axvline(0, c="k", lw=.8); plt.tight_layout(); plt.savefig(out / "figuras/01_eda_quintiles.png", dpi=130); plt.close()

    # ---------------- 3-4. Variables y partición 70/10/20 estratificada
    X = crear_variables(raw)
    y = raw[TARGET]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.125, stratify=y_tmp, random_state=SEED)
    print(f"Train {len(X_tr):,} | Calibración {len(X_cal):,} | Test {len(X_te):,}")

    num = [c for c in X.columns if c not in CAT]
    pre_lin = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pre_spl = ColumnTransformer([("n", Pipeline([("s", StandardScaler()),
                                                  ("sp", SplineTransformer(n_knots=5, degree=3))]), num),
                                 ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pre_tree = ColumnTransformer([("n", "passthrough", num), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])

    # ---------------- 5. Comparación por CV (orden = complejidad creciente)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    espacio = {"num_leaves": [4, 7, 15], "min_child_samples": [20, 40, 80], "learning_rate": [0.02, 0.03, 0.05],
               "n_estimators": [200, 400, 700], "reg_lambda": [0, 5, 20], "colsample_bytree": [0.6, 0.8, 1.0]}
    lgb_base = lgb.LGBMClassifier(subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1)
    busq = RandomizedSearchCV(lgb_base, espacio, n_iter=15, cv=cv, scoring="neg_log_loss",
                              random_state=SEED, n_jobs=1).fit(X_tr, y_tr)
    print("LightGBM mejores hiperparámetros:", busq.best_params_)

    modelos = {
        "1. Baseline (tasa global)": Pipeline([("p", pre_lin), ("m", DummyClassifier(strategy="prior"))]),
        "2. Regresión logística": Pipeline([("p", pre_lin), ("m", LogisticRegression(C=1.0, max_iter=3000))]),
        "3. Logística + splines": Pipeline([("p", pre_spl), ("m", LogisticRegression(C=0.3, max_iter=5000))]),
        "4. Random Forest": Pipeline([("p", pre_tree), ("m", RandomForestClassifier(
            400, min_samples_leaf=20, n_jobs=-1, random_state=SEED))]),
        "5. XGBoost": Pipeline([("p", pre_tree), ("m", xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, random_state=SEED, n_jobs=-1))]),
        "6. LightGBM (ajustado)": busq.best_estimator_,
    }
    scoring = {"auc": "roc_auc", "pr": "average_precision", "ll": "neg_log_loss", "brier": "neg_brier_score"}
    filas = []
    for nombre, mod in modelos.items():
        r = cross_validate(mod, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        filas.append({"modelo": nombre, "AUC": r["test_auc"].mean(), "AUC_std": r["test_auc"].std(),
                      "PR_AUC": r["test_pr"].mean(), "LogLoss": -r["test_ll"].mean(),
                      "LogLoss_std": r["test_ll"].std(), "Brier": -r["test_brier"].mean()})
        print(f"  {nombre:28s} AUC={filas[-1]['AUC']:.4f} (±{filas[-1]['AUC_std']:.4f})  "
              f"LogLoss={filas[-1]['LogLoss']:.4f}")
    comp = pd.DataFrame(filas)
    comp.round(4).to_csv(out / "comparacion_modelos_cv.csv", index=False)

    # Regla de parsimonia: entre los modelos cuyo log-loss en CV está a menos de TOL_LOGLOSS del mejor
    # (diferencia sin relevancia práctica: ~0.001 de AUC), se elige el MÁS SIMPLE.
    # Razón de negocio: explicabilidad (odds ratios, motivos por empresa) y despliegue sencillo, sin perder desempeño.
    mejor_ll = comp["LogLoss"].min()
    candidatos = comp[(comp["LogLoss"] <= mejor_ll + TOL_LOGLOSS) & (~comp["modelo"].str.startswith("1."))]
    elegido_nombre = candidatos.iloc[0]["modelo"]  # comp está ordenado de simple a complejo
    print(f"\nModelo elegido por parsimonia (tolerancia {TOL_LOGLOSS} en log-loss): {elegido_nombre}")
    final = modelos[elegido_nombre].fit(X_tr, y_tr)
    es_logit = elegido_nombre.startswith(("2.", "3."))

    # Calibración: la logística ya estima probabilidades; árboles se calibran (Platt) con el set de calibración.
    if not es_logit:
        final = CalibratedClassifierCV(FrozenEstimator(final), method="sigmoid").fit(X_cal, y_cal)

    # ---------------- 6. Evaluación final en test (una sola vez)
    p_te = final.predict_proba(X_te)[:, 1]
    tabla = pd.DataFrame({
        "Modelo final": metricas(y_te, p_te),
        "Logística (referencia)": metricas(y_te, modelos["2. Regresión logística"].fit(X_tr, y_tr).predict_proba(X_te)[:, 1]),
        "LightGBM": metricas(y_te, modelos["6. LightGBM (ajustado)"].fit(X_tr, y_tr).predict_proba(X_te)[:, 1]),
        "Baseline": metricas(y_te, modelos["1. Baseline (tasa global)"].fit(X_tr, y_tr).predict_proba(X_te)[:, 1]),
    }).T
    tabla.round(4).to_csv(out / "metricas_test.csv")
    print("\nMétricas en TEST:\n", tabla.round(4).to_string())

    # IC 95% de AUC por bootstrap
    rng = np.random.default_rng(SEED)
    yt, pt = y_te.values, p_te
    aucs = [roc_auc_score(yt[i], pt[i]) for i in (rng.integers(0, len(yt), len(yt)) for _ in range(500))]
    auc_ic = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    print(f"AUC test {roc_auc_score(yt, pt):.4f}  IC95% [{auc_ic[0]:.4f}, {auc_ic[1]:.4f}]")

    # Desempeño por segmento (¿el modelo falla en algún grupo?)
    seg = X_te.assign(y=y_te.values, p=p_te)
    seg["tamano"] = pd.cut(seg["num_trabajadores"], [0, 15, 40, 1e9], labels=["5-15", "16-40", ">40"])
    filas = []
    for c in ["sector", "tamano"]:
        for k, g in seg.groupby(c, observed=True):
            if g["y"].nunique() == 2:
                filas.append({"segmento": c, "grupo": k, "n": len(g), "tasa_real": g["y"].mean(),
                              "prob_media": g["p"].mean(), "AUC": roc_auc_score(g["y"], g["p"])})
    pd.DataFrame(filas).round(3).to_csv(out / "desempeno_por_segmento.csv", index=False)

    # Figuras: ROC / PR / calibración
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    fpr, tpr, _ = roc_curve(y_te, p_te); ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set(title=f"ROC (AUC={roc_auc_score(y_te, p_te):.3f})", xlabel="FPR", ylabel="TPR")
    pr, rc, _ = precision_recall_curve(y_te, p_te); ax[1].plot(rc, pr); ax[1].axhline(tasa, c="k", ls="--")
    ax[1].set(title=f"Precisión-Recall (AP={average_precision_score(y_te, p_te):.3f})", xlabel="Recall", ylabel="Precisión")
    fr, md = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    ax[2].plot(md, fr, "o-"); ax[2].plot([0, 1], [0, 1], "k--")
    ax[2].set(title="Calibración", xlabel="Prob. predicha", ylabel="Frecuencia observada")
    plt.tight_layout(); plt.savefig(out / "figuras/02_roc_pr_calibracion.png", dpi=130); plt.close()

    # ---------------- 7. Valor de negocio
    g = tabla_ganancias(y_te, p_te)
    g.round(3).to_csv(out / "ganancias_por_decil.csv")
    print("\nLift por decil (decil 1 = mayor riesgo):\n", g[["tasa_real", "lift", "captura_acum_%"]].round(2).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(g.index, g["lift"]); ax[0].axhline(1, c="k", ls="--"); ax[0].set(title="Lift por decil de riesgo", xlabel="Decil")
    ax[1].plot([0] + list(g.index * 10), [0] + list(g["captura_acum_%"]), "o-", label="Modelo")
    ax[1].plot([0, 100], [0, 100], "k--", label="Al azar"); ax[1].legend()
    ax[1].set(title="% de abandonos capturados vs. % de empresas contactadas", xlabel="% empresas contactadas")
    plt.tight_layout(); plt.savefig(out / "figuras/03_lift_ganancias.png", dpi=130); plt.close()

    # Decisión: valor esperado de contactar a la empresa i = p_i * efectividad * valor - costo.
    # Regla: contactar si p_i > costo / (efectividad * valor). Se compara contra "no hacer nada" y "contactar a todos".
    corte_ev = COSTO_ACCION / (EFECTIVIDAD * VALOR_EMPRESA_RETENIDA)
    orden = np.argsort(-p_te)
    ev_acum = np.cumsum(p_te[orden] * EFECTIVIDAD * VALOR_EMPRESA_RETENIDA - COSTO_ACCION)  # con prob. calibrada
    ev_real = np.cumsum(y_te.values[orden] * EFECTIVIDAD * VALOR_EMPRESA_RETENIDA - COSTO_ACCION)  # con abandono real
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, len(ev_real) + 1) / len(ev_real) * 100, ev_real / 1e6, label="Beneficio neto (resultado real en test)")
    plt.axvline(100 * (p_te >= corte_ev).mean(), c="r", ls="--", label=f"Regla p ≥ {corte_ev:.0%}")
    plt.xlabel("% de empresas contactadas (de mayor a menor riesgo)"); plt.ylabel("Millones COP (test)")
    plt.title("Beneficio neto de la campaña de retención (supuestos ilustrativos)"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "figuras/04_beneficio_campana.png", dpi=130); plt.close()
    contactar_a_todos = (y_te.values * EFECTIVIDAD * VALOR_EMPRESA_RETENIDA - COSTO_ACCION).sum()
    n_regla = int((p_te >= corte_ev).sum())
    beneficio_regla = float(ev_real[n_regla - 1]) if n_regla else 0.0
    # Sensibilidad: el punto de corte depende de supuestos -> mostrar el rango
    sens = []
    for ef in [0.10, 0.20, 0.30]:
        for val in [10e6, 20e6, 40e6]:
            c = COSTO_ACCION / (ef * val)
            sens.append({"efectividad": ef, "valor_empresa_M": val / 1e6, "corte_prob": round(c, 3),
                         "%_empresas_a_contactar": round(100 * (p_te >= c).mean(), 1)})
    pd.DataFrame(sens).to_csv(out / "sensibilidad_corte.csv", index=False)
    print(f"\nRegla de decisión: contactar si p ≥ {corte_ev:.1%} -> {n_regla/len(p_te):.1%} de empresas | "
          f"beneficio neto test {beneficio_regla/1e6:,.0f} M COP vs {contactar_a_todos/1e6:,.0f} M contactando a todos")

    # Niveles de riesgo operativos (percentiles sobre calibración -> estables)
    p_cal = final.predict_proba(X_cal)[:, 1]
    cortes = {"alto": float(np.quantile(p_cal, 0.80)), "medio": float(np.quantile(p_cal, 0.50))}
    nivel = np.where(p_te >= cortes["alto"], "Alto", np.where(p_te >= cortes["medio"], "Medio", "Bajo"))
    pd.DataFrame({"nivel": nivel, "y": y_te.values}).groupby("nivel")["y"].agg(["count", "mean"]).round(3) \
        .to_csv(out / "niveles_riesgo_test.csv")

    # ---------------- 8. Factores asociados
    # 8a. Logit con statsmodels sobre variables estandarizadas: odds ratio por +1 desv. estándar (comparables entre sí).
    # Se excluyen ln_trabajadores y ratio_afiliacion (derivadas) para no inducir colinealidad.
    base_cols = [c for c in num_raw]
    Z = X_tr[base_cols].copy(); Z = (Z - Z.mean()) / Z.std()
    Z = pd.concat([Z, pd.get_dummies(X_tr["sector"], drop_first=True).astype(float)], axis=1)
    logit = sm.Logit(y_tr, sm.add_constant(Z)).fit(disp=0, cov_type="HC3")
    ci = logit.conf_int()
    odds = pd.DataFrame({"odds_ratio": np.exp(logit.params), "IC95_inf": np.exp(ci[0]),
                         "IC95_sup": np.exp(ci[1]), "p_valor": logit.pvalues}).drop(index="const")
    odds.round(4).sort_values("odds_ratio").to_csv(out / "odds_ratios_logit.csv")
    vif = pd.Series([variance_inflation_factor(Z[base_cols].values, i) for i in range(len(base_cols))], base_cols)
    vif.round(2).to_csv(out / "vif.csv")
    print(f"\nVIF máximo: {vif.max():.2f} (>5 sería preocupante)")
    o = odds.sort_values("odds_ratio")
    plt.figure(figsize=(7, 7))
    plt.errorbar(o["odds_ratio"], range(len(o)), xerr=[o["odds_ratio"] - o["IC95_inf"], o["IC95_sup"] - o["odds_ratio"]], fmt="o")
    plt.yticks(range(len(o)), o.index); plt.axvline(1, c="k", ls="--"); plt.xscale("log")
    plt.title("Odds ratio de abandono por +1 desv. estándar (IC 95%)\n<1 protege · >1 aumenta el riesgo")
    plt.tight_layout(); plt.savefig(out / "figuras/05_odds_ratios.png", dpi=130); plt.close()

    # 8b. Permutation importance en test, medida como caída de AUC (importancia honesta del modelo final)
    pi = permutation_importance(final, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, X.columns).sort_values()
    imp.round(5).to_csv(out / "permutation_importance.csv")
    imp.plot.barh(figsize=(7, 6), title="Permutation importance (caída de AUC en test)")
    plt.tight_layout(); plt.savefig(out / "figuras/06_permutation_importance.png", dpi=130); plt.close()

    # 8c. SHAP sobre LightGBM (forma y dirección de cada efecto, incluyendo posibles no linealidades)
    gbm = modelos["6. LightGBM (ajustado)"]
    sv = shap.TreeExplainer(gbm).shap_values(X_te)
    sv = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(sv, X_te, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(out / "figuras/07_shap_summary.png", dpi=130); plt.close()

    # ---------------- 9. Persistencia
    joblib.dump({"modelo": final, "es_logit": es_logit, "nombre": elegido_nombre, "columnas": list(X.columns),
                 "categorias": list(X["sector"].cat.categories), "cortes_riesgo": cortes,
                 "num_cols": num, "corte_ev": corte_ev}, out / "modelo_desafiliacion.joblib")
    resumen = {"modelo": elegido_nombre, "tasa_abandono": float(tasa), "auc_test": float(roc_auc_score(yt, pt)),
               "auc_ic95": auc_ic, "cv": comp.round(4).to_dict("records"), "test": tabla.round(4).to_dict("index"),
               "corte_ev": corte_ev, "pct_contactar_regla": n_regla / len(p_te),
               "beneficio_regla_M": beneficio_regla / 1e6, "beneficio_todos_M": float(contactar_a_todos) / 1e6,
               "cortes_riesgo": cortes,
               "supuestos_economicos": {"valor": VALOR_EMPRESA_RETENIDA, "costo": COSTO_ACCION, "efectividad": EFECTIVIDAD}}
    (out / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nArtefactos guardados en {out.resolve()}")


# %% ------------------------------------------------------------------ scoring
def predecir(df_nuevas: pd.DataFrame, ruta_modelo: str = "outputs_reto2/modelo_desafiliacion.joblib") -> pd.DataFrame:
    """Scoring de empresas (mismas columnas que el CSV, sin 'abandono').
    Devuelve probabilidad, nivel de riesgo y, si el modelo es logístico, los 3 factores que más empujan el riesgo."""
    art = joblib.load(ruta_modelo)
    X = crear_variables(df_nuevas)[art["columnas"]]
    X["sector"] = pd.Categorical(X["sector"].astype(str), categories=art["categorias"])
    p = art["modelo"].predict_proba(X)[:, 1]
    c = art["cortes_riesgo"]
    res = pd.DataFrame({"id_empresa": df_nuevas.get("id_empresa"), "prob_desafiliacion": p,
                        "nivel_riesgo": np.where(p >= c["alto"], "Alto", np.where(p >= c["medio"], "Medio", "Bajo")),
                        "contactar_segun_costo_beneficio": p >= art["corte_ev"]})
    if art["es_logit"]:
        pipe = art["modelo"]
        if "3." in art["nombre"]:
            pass  # con splines los coeficientes no son por variable: se omite el desglose
        else:
            Zs = pipe.named_steps["p"].transform(X)
            nombres = pipe.named_steps["p"].get_feature_names_out()
            contrib = Zs.toarray() if hasattr(Zs, "toarray") else Zs
            contrib = contrib * pipe.named_steps["m"].coef_[0]
            top = np.argsort(-contrib, axis=1)[:, :3]
            res["factores_principales"] = [", ".join(nombres[j].split("__")[1] for j in fila) for fila in top]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="base_clasificacion.csv")
    ap.add_argument("--out", default="outputs_reto2")
    a = ap.parse_args()
    main(a.data, a.out)
