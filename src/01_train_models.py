"""
01_train_models.py — Treina e compara os cinco modelos ensemble.

Lê alocacoes.csv, treina Random Forest, XGBoost, LightGBM, CatBoost e
Extra Trees, avalia cada um (R², MAE, RMSE, validação cruzada) e salva
os resultados em outputs/ml_resultados.pkl para uso pelos demais scripts.

Referências dos métodos utilizados:
    Random Forest  — Breiman (2001)
    Extra Trees    — Geurts, Ernst & Wehenkel (2006)
    Gradient Boost — Friedman (2001, 2002)
    XGBoost        — Chen & Guestrin (2016)
    LightGBM       — Ke et al. (2017)
    CatBoost       — Prokhorenkova et al. (2018)
    Validação cruzada — Kohavi (1995)

Uso:
    python src/01_train_models.py
"""
from __future__ import annotations
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import config as C

warnings.filterwarnings("ignore")


def build_models() -> dict:
    """Instancia os cinco modelos com hiperparâmetros do artigo."""
    rs = C.RANDOM_STATE
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=C.N_ESTIMATORS, max_features="sqrt",
            min_samples_leaf=2, n_jobs=-1, random_state=rs),
        "XGBoost": XGBRegressor(
            n_estimators=C.N_ESTIMATORS, learning_rate=C.LEARNING_RATE,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            random_state=rs, verbosity=0),
        "LightGBM": LGBMRegressor(
            n_estimators=C.N_ESTIMATORS, learning_rate=C.LEARNING_RATE,
            max_depth=6, num_leaves=31, subsample=0.8,
            random_state=rs, verbose=-1),
        "CatBoost": CatBoostRegressor(
            iterations=C.N_ESTIMATORS, learning_rate=C.LEARNING_RATE,
            depth=6, random_state=rs, verbose=0, allow_writing_files=False),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=C.N_ESTIMATORS, min_samples_leaf=2,
            n_jobs=-1, random_state=rs),
    }


def main() -> None:
    C.require_data()
    df = pd.read_csv(C.ALOCACOES_CSV)

    features = [c for c in df.columns if c not in C.ID_COLUMNS + [C.TARGET]]
    X, y = df[features], df[C.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    kf = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    resultados, ajustados = {}, {}
    header = f"{'Modelo':<16}{'R²':>10}{'CV R²':>10}{'CV Std':>10}{'MAE':>10}{'RMSE':>10}"
    print(header)
    print("-" * len(header))

    for nome, modelo in build_models().items():
        cv = cross_val_score(modelo, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        modelo.fit(X_train, y_train)
        y_pred = np.clip(modelo.predict(X_test), 0, 1)
        resultados[nome] = {
            "r2":   r2_score(y_test, y_pred),
            "cv_mean": cv.mean(), "cv_std": cv.std(),
            "mae":  mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        ajustados[nome] = modelo
        r = resultados[nome]
        print(f"{nome:<16}{r['r2']:>10.4f}{r['cv_mean']:>10.4f}"
              f"{r['cv_std']:>10.4f}{r['mae']:>10.4f}{r['rmse']:>10.4f}")

    melhor = max(resultados, key=lambda k: resultados[k]["r2"])
    print(f"\nMelhor modelo: {melhor} (R²={resultados[melhor]['r2']:.4f})")

    # Importância de features do melhor modelo
    mb = ajustados[melhor]
    if hasattr(mb, "feature_importances_"):
        imp = mb.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(
            mb, X_test, y_test, n_repeats=10,
            random_state=C.RANDOM_STATE).importances_mean
    fi = (pd.DataFrame({"feature": features, "importance": imp})
          .sort_values("importance", ascending=False))

    with open(C.RESULTS_PKL, "wb") as f:
        pickle.dump({
            "resultados": resultados,
            "melhor": melhor,
            "fi": fi,
            "y_test": y_test.values,
            "y_pred": np.clip(ajustados[melhor].predict(X_test), 0, 1),
            "features": features,
        }, f)
    print(f"\nResultados salvos em {C.RESULTS_PKL.relative_to(C.ROOT_DIR)}")


if __name__ == "__main__":
    main()
