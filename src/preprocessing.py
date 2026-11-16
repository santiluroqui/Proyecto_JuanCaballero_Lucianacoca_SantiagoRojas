"""Funciones reutilizables de carga, limpieza, transformación y PCA.

Cada función hace una sola cosa y devuelve también un registro de lo que quitó,
para poder documentar en el informe cuántas filas se descartaron y por qué.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config as cfg


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def load_raw(path=None) -> pd.DataFrame:
    """Lee el CSV crudo. Falla con un mensaje claro si no está descargado."""
    path = path or cfg.RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}.\n"
            "Descarga dataset.csv desde "
            "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset "
            "y colócalo en data/raw/ (ver data/raw/README.md)."
        )
    df = pd.read_csv(path)
    # El CSV trae un índice exportado como columna sin nombre.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df


# --------------------------------------------------------------------------- #
# Limpieza
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia el dataset y devuelve (df_limpio, bitácora_de_limpieza).

    Pasos (cada uno queda contado en la bitácora):
      1. Quitar filas con nulos en variables del modelo o en identificadores.
      2. Quitar registros inválidos (tempo = 0, duración = 0): son pistas
         vacías/silencio, no música medible.
      3. Consolidar duplicados por track_id (una canción listada en varios
         géneros): se conserva el primero y se guarda n_genres.
      4. Quitar duplicados por (artista, nombre de canción), que suelen ser la
         misma canción en álbumes distintos con otro track_id.
    """
    log = []

    def step(name, before, after):
        log.append({"paso": name, "filas_antes": before, "filas_despues": after,
                    "descartadas": before - after})

    out = df.copy()
    n0 = len(out)

    # 1. nulos
    needed = cfg.FEATURES + [cfg.ID_COL, cfg.GENRE_COL]
    before = len(out)
    out = out.dropna(subset=needed)
    step("nulos en variables/identificadores", before, len(out))

    # 2. registros inválidos
    before = len(out)
    out = out[(out["tempo"] > 0) & (out["duration_ms"] > 0)]
    step("tempo = 0 o duración = 0", before, len(out))

    # 3. duplicados por track_id (multigénero)
    genres_per_track = out.groupby(cfg.ID_COL)[cfg.GENRE_COL].nunique().rename("n_genres")
    before = len(out)
    out = out.drop_duplicates(subset=cfg.ID_COL, keep="first")
    out = out.merge(genres_per_track, on=cfg.ID_COL, how="left")
    step("duplicados por track_id", before, len(out))

    # 4. duplicados por artista + nombre
    if cfg.DEDUP_BY_NAME and {"artists", "track_name"} <= set(out.columns):
        before = len(out)
        out = out.drop_duplicates(subset=["artists", "track_name"], keep="first")
        step("duplicados por artista + nombre", before, len(out))

    out = out.reset_index(drop=True)
    log.append({"paso": "TOTAL", "filas_antes": n0, "filas_despues": len(out),
                "descartadas": n0 - len(out)})
    return out, pd.DataFrame(log)


# --------------------------------------------------------------------------- #
# Outliers (solo se REPORTAN; no se eliminan, se justifica en el notebook)
# --------------------------------------------------------------------------- #
def iqr_outlier_report(df: pd.DataFrame, features=None) -> pd.DataFrame:
    features = features or cfg.FEATURES
    rows = []
    for f in features:
        q1, q3 = df[f].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[f] < lo) | (df[f] > hi)).sum())
        rows.append({"variable": f, "q1": q1, "q3": q3, "limite_inf": lo, "limite_sup": hi,
                     "n_outliers": n_out, "pct_outliers": 100 * n_out / len(df),
                     "asimetria": df[f].skew()})
    return pd.DataFrame(rows).set_index("variable")


# --------------------------------------------------------------------------- #
# Transformación y escalado
# --------------------------------------------------------------------------- #
def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica log1p a las variables sesgadas y devuelve solo las FEATURES."""
    X = df[cfg.FEATURES].astype(float).copy()
    for f in cfg.LOG_FEATURES:
        X[f] = np.log1p(X[f])
    return X


def scale(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return Xs, scaler


def stratified_sample(df: pd.DataFrame, n: int | None = None) -> pd.Series:
    """Booleano in_sample: muestra estratificada por género (proporcional)."""
    n = min(n or cfg.SAMPLE_SIZE, len(df))
    frac = n / len(df)
    idx = (df.groupby(cfg.GENRE_COL, group_keys=False)
             .apply(lambda g: g.sample(frac=frac, random_state=cfg.RANDOM_STATE),
                    include_groups=False).index)
    flag = pd.Series(False, index=df.index, name="in_sample")
    flag.loc[idx] = True
    return flag


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def fit_pca(Xs: pd.DataFrame, variance_target: float | None = None):
    """Ajusta PCA con todos los componentes y elige cuántos conservar.

    Devuelve (pca_completo, n_componentes_elegidos, tabla_varianza, scores,
    loadings). Los loadings son correlaciones variable-componente
    (componentes * sqrt(autovalor)), más fáciles de interpretar que los pesos.
    """
    variance_target = variance_target or cfg.PCA_VARIANCE_TARGET
    pca = PCA(random_state=cfg.RANDOM_STATE).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, variance_target) + 1)

    pcs = [f"PC{i + 1}" for i in range(Xs.shape[1])]
    var_table = pd.DataFrame({
        "componente": pcs,
        "autovalor": pca.explained_variance_,
        "varianza_explicada": pca.explained_variance_ratio_,
        "varianza_acumulada": cum,
    })
    scores = pd.DataFrame(pca.transform(Xs)[:, :k], columns=pcs[:k], index=Xs.index)
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),
                            index=Xs.columns, columns=pcs)
    return pca, k, var_table, scores, loadings


# --------------------------------------------------------------------------- #
# Guardado y carga de entregables (ver HANDOFF_JUAN.md)
# --------------------------------------------------------------------------- #
def _update_config_json(**kwargs):
    path = cfg.DATA_PROCESSED / "config_preprocesamiento.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(kwargs)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_preprocessed(clean_df, Xs, scaler, in_sample):
    """Etapa 2 (preprocesamiento): songs_clean.csv, X_scaled.csv y scaler."""
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ids = clean_df[[cfg.ID_COL]]

    # Metadatos + género + variables originales (para interpretar clusters)
    meta_cols = [c for c in cfg.META_COLS if c in clean_df.columns]
    meta = clean_df[meta_cols + [cfg.GENRE_COL, "n_genres"] + cfg.FEATURES].copy()
    meta = meta.rename(columns={cfg.GENRE_COL: "primary_genre"})
    meta["in_sample"] = in_sample.values
    meta.to_csv(cfg.DATA_PROCESSED / "songs_clean.csv", index=False)

    # Matriz para clustering: SIN género, SIN metadatos
    pd.concat([ids, Xs], axis=1).to_csv(cfg.DATA_PROCESSED / "X_scaled.csv", index=False)
    joblib.dump(scaler, cfg.DATA_PROCESSED / "scaler.joblib")
    _update_config_json(
        features=cfg.FEATURES, log_features=cfg.LOG_FEATURES,
        n_filas=int(len(clean_df)), n_filas_muestra=int(in_sample.sum()),
        random_state=cfg.RANDOM_STATE,
    )


def load_preprocessed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (songs_clean, X_scaled) alineados por fila."""
    songs = pd.read_csv(cfg.DATA_PROCESSED / "songs_clean.csv")
    Xs = pd.read_csv(cfg.DATA_PROCESSED / "X_scaled.csv")
    assert (songs[cfg.ID_COL].values == Xs[cfg.ID_COL].values).all(), "filas desalineadas"
    return songs, Xs


def save_pca(ids, pca, k, scores, var_table, loadings):
    """Etapa 3 (PCA): X_pca.csv, pca.joblib y tablas de varianza/loadings."""
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat([ids.reset_index(drop=True), scores.reset_index(drop=True)], axis=1) \
        .to_csv(cfg.DATA_PROCESSED / "X_pca.csv", index=False)
    var_table.to_csv(cfg.TABLES / "pca_varianza_explicada.csv", index=False)
    loadings.to_csv(cfg.TABLES / "pca_loadings.csv")
    joblib.dump(pca, cfg.DATA_PROCESSED / "pca.joblib")
    _update_config_json(
        n_componentes_pca=int(k),
        varianza_acumulada_pca=float(var_table["varianza_acumulada"].iloc[k - 1]),
        objetivo_varianza=cfg.PCA_VARIANCE_TARGET,
    )
