"""Ejecuta de punta a punta los Módulos 2 y 3 sin abrir notebooks.

Uso (desde la raíz del repo):
    python scripts/run_pipeline.py

Genera todo lo que Juan necesita en data/processed/. Es el mismo flujo que
documentan los notebooks 02 y 03; sirve para reproducir los resultados con un
solo comando.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src import preprocessing as pp


def main():
    raw = pp.load_raw()
    print(f"Crudo: {raw.shape[0]:,} filas x {raw.shape[1]} columnas")

    df, log = pp.clean(raw)
    print(log.to_string(index=False))
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    log.to_csv(cfg.TABLES / "bitacora_limpieza.csv", index=False)

    X = pp.transform_features(df)
    Xs, scaler = pp.scale(X)
    in_sample = pp.stratified_sample(df)
    pp.save_preprocessed(df, Xs, scaler, in_sample)
    print(f"Muestra estratificada: {int(in_sample.sum()):,} canciones")

    pca, k, var_table, scores, loadings = pp.fit_pca(Xs)
    pp.save_pca(df[[cfg.ID_COL]], pca, k, scores, var_table, loadings)
    acc = var_table["varianza_acumulada"].iloc[k - 1]
    print(f"PCA: {k} componentes explican {acc:.1%} de la varianza")
    print(f"Listo. Entregables en {cfg.DATA_PROCESSED}")


if __name__ == "__main__":
    main()
