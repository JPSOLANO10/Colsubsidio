# %% [markdown]
# # Reto 1 - Estimación del aporte mensual de empresas afiliadas (Colsubsidio)
#
# Ejecución:  python reto1_aporte_mensual.py --data empresas_afiliadas.csv --out outputs
#
# Flujo:
#   1. Carga y validación de calidad de datos
#   2. EDA orientado a decisiones (distribución del objetivo, relaciones clave)
#   3. Ingeniería de variables (con justificación de negocio)
#   4. Partición train / calibración / test (sin fuga de información)
#   5. Comparación de modelos con validación cruzada
#   6. Ajuste del mejor modelo + evaluación final en test
#   7. Intervalos de predicción (conformal) y error por segmento
#   8. Interpretabilidad: elasticidades (OLS log-log), permutation importance y SHAP
#   9. Persistencia del modelo y función de scoring para producción
#
# SUPUESTOS (también se documentan en el informe):
#   S1. Cada fila es una empresa única y 'aporte_mensual' está en COP del mismo período.
#   S2. No hay dimensión temporal en el archivo -> el modelo es transversal (no un pronóstico en el tiempo).
#   S3. 'salario_promedio' se interpreta en millones de COP (escala observada 1.5-7).
#   S4. 'satisfaccion_servicio' y 'numero_servicios' se asumen disponibles al momento de predecir;
#       si se miden DESPUÉS del aporte, habría que excluirlas (riesgo de fuga / causalidad inversa).
#   S5. Los valores extremos (empresas grandes) son reales, no errores: se conservan.
# %%
import argparse
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # permite correr sin pantalla (servidor / CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_val_predict, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "aporte_mensual"
CAT = ["ciudad", "sector"]


# %% ------------------------------------------------------------------ utilidades
def cargar_y_validar(path: str) -> pd.DataFrame:
    """Carga el CSV (utf-8-sig por el BOM) y corre chequeos de calidad."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert df["id_empresa"].is_unique, "id_empresa duplicado"
    assert df.isna().sum().sum() == 0, "Hay nulos: definir estrategia de imputación"
    assert (df[TARGET] > 0).all(), "Aporte <= 0: no se puede usar log"
    assert (df["numero_trabajadores"] > 0).all()
    return df


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de variables. Función pura -> la misma en entrenamiento y producción.

    - ln_trabajadores: el aporte escala ~multiplicativamente con el tamaño; el log lo linealiza.
    - ln_nomina_total: proxy de masa salarial (trabajadores x salario), base natural del aporte.
    - aporte por trabajador NO se crea: usaría el objetivo (fuga).
    """
    out = df.copy()
    out["ln_trabajadores"] = np.log(out["numero_trabajadores"])
    out["ln_nomina_total"] = np.log(out["numero_trabajadores"] * out["salario_promedio"])
    out["tiene_servicios"] = (out["numero_servicios"] > 0).astype(int)
    for c in CAT:
        out[c] = out[c].astype("category")
    return out


def metricas(y_log_real, y_log_pred, smearing=1.0) -> dict:
    """Métricas en escala log (donde se entrena) y en COP (donde decide el negocio)."""
    real, pred = np.exp(y_log_real), np.exp(y_log_pred) * smearing
    return {
        "R2_log": r2_score(y_log_real, y_log_pred),
        "RMSE_log": mean_squared_error(y_log_real, y_log_pred) ** 0.5,
        "MAE_COP": mean_absolute_error(real, pred),
        "MAPE": mean_absolute_percentage_error(real, pred),
        "WAPE": np.abs(real - pred).sum() / real.sum(),   # error ponderado por monto
        "R2_COP": r2_score(real, pred),
        "Sesgo_total_%": 100 * (pred.sum() / real.sum() - 1),  # clave para presupuesto
    }


# %% ------------------------------------------------------------------ main
def main(data_path: str, out_dir: str):
    out = Path(out_dir)
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ---------------- 1. Carga
    raw = cargar_y_validar(data_path)
    print(f"Datos: {raw.shape[0]:,} empresas, {raw.shape[1]} columnas, sin nulos ni duplicados")

    # ---------------- 2. EDA (figuras para el informe)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(raw[TARGET] / 1e6, bins=60, ax=ax[0]).set(title="Aporte mensual (millones COP)")
    sns.histplot(np.log(raw[TARGET]), bins=60, ax=ax[1]).set(title="ln(aporte mensual)")
    plt.tight_layout(); plt.savefig(out / "figuras/01_objetivo.png", dpi=130); plt.close()
    print(f"Asimetría objetivo: {raw[TARGET].skew():.2f} -> log: {np.log(raw[TARGET]).skew():.2f}")

    num_cols = raw.select_dtypes("number").drop(columns=["id_empresa"]).columns
    corr = raw[num_cols].corr(method="spearman")
    plt.figure(figsize=(9, 7)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Correlación de Spearman (robusta a asimetría)")
    plt.tight_layout(); plt.savefig(out / "figuras/02_correlaciones.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, c in zip(ax, CAT):
        orden = raw.groupby(c)[TARGET].median().sort_values().index
        sns.boxplot(data=raw, y=c, x=TARGET, order=orden, ax=a, showfliers=False)
        a.set_title(f"Aporte por {c} (sin outliers)"); a.set_xlabel("COP")
    plt.tight_layout(); plt.savefig(out / "figuras/03_aporte_por_categoria.png", dpi=130); plt.close()

    # ---------------- 3-4. Variables y partición: 70% train / 10% calibración / 20% test
    df = crear_variables(raw)
    y = np.log(df[TARGET])
    X = df.drop(columns=["id_empresa", TARGET])
    # Estratificamos por deciles del objetivo para que train/test tengan la misma cola
    estrato = pd.qcut(y, 10, labels=False)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=estrato)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.125, random_state=SEED,
                                                stratify=estrato.loc[X_tmp.index])
    print(f"Train {len(X_tr):,} | Calibración {len(X_cal):,} | Test {len(X_te):,}")

    num = [c for c in X.columns if c not in CAT]
    # Preprocesamiento lineal: escalar + one-hot. Árboles: categorías nativas.
    pre_lin = ColumnTransformer([("n", StandardScaler(), num),
                                 ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pre_tree = ColumnTransformer([("n", "passthrough", num),
                                  ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])

    # ---------------- 5. Comparación de modelos (CV 5 folds sobre train)
    modelos = {
        "Baseline (mediana)": Pipeline([("p", pre_lin), ("m", DummyRegressor(strategy="median"))]),
        "Ridge (log-lineal)": Pipeline([("p", pre_lin), ("m", RidgeCV(alphas=np.logspace(-3, 3, 20)))]),
        "Random Forest": Pipeline([("p", pre_tree), ("m", RandomForestRegressor(
            300, min_samples_leaf=5, n_jobs=-1, random_state=SEED))]),
        "XGBoost": Pipeline([("p", pre_tree), ("m", xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1))]),
        "LightGBM": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                                       min_child_samples=30, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, random_state=SEED, verbose=-1),
    }
    cv = KFold(5, shuffle=True, random_state=SEED)
    filas = []
    for nombre, mod in modelos.items():
        r = cross_validate(mod, X_tr, y_tr, cv=cv, n_jobs=1,
                           scoring=("neg_root_mean_squared_error", "r2"))
        filas.append({"modelo": nombre, "RMSE_log_cv": -r["test_neg_root_mean_squared_error"].mean(),
                      "std": r["test_neg_root_mean_squared_error"].std(), "R2_log_cv": r["test_r2"].mean()})
        print(f"  {nombre:20s} RMSE_log={filas[-1]['RMSE_log_cv']:.4f} (±{filas[-1]['std']:.4f})")
    comp = pd.DataFrame(filas).sort_values("RMSE_log_cv")
    comp.to_csv(out / "comparacion_modelos_cv.csv", index=False)

    # ---------------- 6. Ajuste del LightGBM con búsqueda aleatoria
    espacio = {"num_leaves": [7, 15, 31], "min_child_samples": [10, 20, 40, 80],
               "learning_rate": [0.02, 0.03, 0.05], "n_estimators": [400, 700, 1000],
               "reg_lambda": [0, 1, 5, 10], "colsample_bytree": [0.6, 0.8, 1.0]}
    busq = RandomizedSearchCV(modelos["LightGBM"], espacio, n_iter=15, cv=cv, random_state=SEED,
                              scoring="neg_root_mean_squared_error", n_jobs=1).fit(X_tr, y_tr)
    final = busq.best_estimator_
    print("Mejores hiperparámetros:", busq.best_params_)

    # Corrección de sesgo al volver de log a COP (Duan smearing): exp(pred) subestima la media.
    # Se estima con residuos de CALIBRACIÓN (datos no usados para ajustar).
    res_cal = y_cal - final.predict(X_cal)
    smearing = float(np.mean(np.exp(res_cal)))
    print(f"Factor smearing: {smearing:.4f}")

    # Evaluación final UNA sola vez sobre test
    p_te = final.predict(X_te)
    tabla = pd.DataFrame({
        "LightGBM final": metricas(y_te, p_te, smearing),
        "Ridge (referencia)": metricas(y_te, modelos["Ridge (log-lineal)"].fit(X_tr, y_tr).predict(X_te)),
        "Baseline": metricas(y_te, modelos["Baseline (mediana)"].fit(X_tr, y_tr).predict(X_te)),
    }).T
    tabla.to_csv(out / "metricas_test.csv")
    print("\nMétricas en TEST:\n", tabla.round(3).to_string())

    plt.figure(figsize=(5.5, 5.5))
    real, pred = np.exp(y_te) / 1e6, np.exp(p_te) * smearing / 1e6
    plt.scatter(real, pred, s=6, alpha=.4); lim = [real.min(), real.max()]
    plt.plot(lim, lim, "r--"); plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Real (M COP)"); plt.ylabel("Predicho (M COP)"); plt.title("Predicho vs real (test)")
    plt.tight_layout(); plt.savefig(out / "figuras/04_pred_vs_real.png", dpi=130); plt.close()

    # ---------------- 7. Intervalos de predicción (split conformal ASIMÉTRICO, cobertura nominal 80%)
    # Los residuos tienen colas pesadas y sesgadas a la derecha (ver sección 7b), por eso se usan
    # cuantiles con signo (10% y 90%) en vez de un +/- simétrico.
    n_cal = len(res_cal)
    q_lo = float(np.quantile(res_cal, 0.10, method="lower"))
    q_hi = float(np.quantile(res_cal, 0.90, method="higher"))
    lo, hi = np.exp(p_te + q_lo) * smearing, np.exp(p_te + q_hi) * smearing
    cobertura = float(((np.exp(y_te) >= lo) & (np.exp(y_te) <= hi)).mean())
    print(f"Intervalo 80% asimétrico: [{(np.exp(q_lo)-1)*100:+.1f}%, {(np.exp(q_hi)-1)*100:+.1f}%] "
          f"| cobertura empírica en test: {cobertura:.1%} (n_cal={n_cal})")

    # ---------------- 7b. Diagnóstico de errores grandes (residuos fuera de muestra, 5-fold sobre TODO)
    # Hallazgo: un grupo pequeño de empresas aporta sistemáticamente ~+40% más de lo que explican
    # las variables disponibles. Se estudia si es predecible (AUC) o si es una variable no observada.
    oof = cross_val_predict(final, X, y, cv=KFold(5, shuffle=True, random_state=SEED))
    d = df.assign(residuo=y - oof)
    atip = d["residuo"] > 0.20
    perfil = d.groupby(atip)[num].mean().T.rename(columns={False: "resto", True: "grupo_atipico"})
    perfil.round(2).to_csv(out / "perfil_grupo_atipico.csv")
    auc = cross_val_score(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, verbose=-1),
                          pd.get_dummies(X, columns=CAT).astype(float), atip.astype(int),
                          cv=5, scoring="roc_auc").mean()
    mape_sin = np.abs(np.exp(d.loc[~atip, "residuo"]) - 1).mean()
    print(f"Empresas con aporte >20% sobre lo esperado: {atip.sum()} ({atip.mean():.1%}) | "
          f"AUC para predecirlas: {auc:.2f} | MAPE del resto: {mape_sin:.1%}")
    resumen_atip = {"n": int(atip.sum()), "pct": float(atip.mean()), "auc_prediccion": float(auc),
                    "residuo_medio_log": float(d.loc[atip, "residuo"].mean()), "mape_resto": float(mape_sin)}

    # Error por segmento: ¿dónde NO conviene confiar en el modelo?
    seg = X_te.copy()
    seg["real"], seg["pred"] = np.exp(y_te), np.exp(p_te) * smearing
    seg["ape"] = (seg.pred - seg.real).abs() / seg.real
    seg["tamano"] = pd.cut(seg.numero_trabajadores, [0, 10, 25, 50, 1e9],
                           labels=["1-10", "11-25", "26-50", ">50"])
    por_seg = pd.concat({c: seg.groupby(c, observed=True).ape.agg(["mean", "count"])
                         for c in ["tamano", "sector", "ciudad"]}).round(3)
    por_seg.to_csv(out / "error_por_segmento.csv")

    # ---------------- 8. Interpretabilidad
    # 8a. OLS log-log con errores robustos (HC3): coeficientes legibles como % de cambio.
    Xo = pd.get_dummies(X_tr[[c for c in X.columns if c not in ["ln_nomina_total", "tiene_servicios", "ln_trabajadores"]]],
                        columns=CAT, drop_first=True).astype(float)
    Xo["numero_trabajadores"] = np.log(Xo["numero_trabajadores"])  # -> elasticidad del tamaño
    Xo["salario_promedio"] = np.log(Xo["salario_promedio"])        # -> elasticidad del salario
    Xo = Xo.rename(columns={"numero_trabajadores": "ln_numero_trabajadores", "salario_promedio": "ln_salario_promedio"})
    ols = sm.OLS(y_tr, sm.add_constant(Xo)).fit(cov_type="HC3")
    coef = pd.DataFrame({"coef": ols.params, "p_valor": ols.pvalues, "efecto_%": (np.exp(ols.params) - 1) * 100})
    coef.round(4).to_csv(out / "ols_loglog_coeficientes.csv")
    print(f"\nOLS log-log: R2 ajustado={ols.rsquared_adj:.3f} (modelo explicativo, no el de producción)")

    # 8b. Permutation importance en test (importancia honesta, no inflada por cardinalidad)
    pi = permutation_importance(final, X_te, y_te, n_repeats=10, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, X.columns).sort_values()
    imp.to_csv(out / "permutation_importance.csv")
    imp.plot.barh(figsize=(7, 6), title="Permutation importance (caída de R² en test)")
    plt.tight_layout(); plt.savefig(out / "figuras/05_permutation_importance.png", dpi=130); plt.close()

    # 8c. SHAP: dirección y forma del efecto
    sv = shap.TreeExplainer(final).shap_values(X_te)
    shap.summary_plot(sv, X_te, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(out / "figuras/06_shap_summary.png", dpi=130); plt.close()

    # ---------------- 9. Persistencia
    joblib.dump({"modelo": final, "smearing": smearing, "q_lo": q_lo, "q_hi": q_hi,
                 "columnas": list(X.columns), "categorias": {c: list(X[c].cat.categories) for c in CAT}},
                out / "modelo_aporte.joblib")
    resumen = {"cv": comp.round(4).to_dict("records"), "test": tabla.round(4).to_dict("index"),
               "cobertura_80": cobertura, "smearing": smearing, "mejores_params": busq.best_params_,
               "intervalo_80_pct": [(np.exp(q_lo) - 1) * 100, (np.exp(q_hi) - 1) * 100],
               "grupo_atipico": resumen_atip}
    (out / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nArtefactos guardados en {out.resolve()}")


# %% ------------------------------------------------------------------ scoring
def predecir(df_nuevas: pd.DataFrame, ruta_modelo: str = "outputs/modelo_aporte.joblib") -> pd.DataFrame:
    """Scoring de empresas nuevas (mismas columnas que el CSV, sin 'aporte_mensual').
    Devuelve estimación puntual e intervalo 80%."""
    art = joblib.load(ruta_modelo)
    X = crear_variables(df_nuevas)[art["columnas"]]
    for c, cats in art["categorias"].items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)  # categoría nueva -> NaN (LightGBM la tolera)
    p = art["modelo"].predict(X)
    s = art["smearing"]
    return pd.DataFrame({"id_empresa": df_nuevas.get("id_empresa"),
                         "aporte_estimado": np.exp(p) * s,
                         "limite_inf_80": np.exp(p + art["q_lo"]) * s,
                         "limite_sup_80": np.exp(p + art["q_hi"]) * s})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="empresas_afiliadas.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    main(a.data, a.out)
