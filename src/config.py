"""Configuración central del proyecto.

Todas las decisiones metodológicas (qué variables se usan, qué se transforma,
tamaño de muestra, semilla) viven aquí para que el equipo las vea y las discuta
en un solo lugar, y para que cualquier cambio quede registrado en Git.
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

RAW_FILE = DATA_RAW / "dataset.csv"

# --- Reproducibilidad --------------------------------------------------------
RANDOM_STATE = 42

# --- Variables ---------------------------------------------------------------
# Variables acústicas que SÍ entran al clustering (10, según el plan).
FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Variables con distribución muy sesgada: se transforman con log1p ANTES de
# estandarizar (se justifica con los histogramas del notebook 01).
LOG_FEATURES = ["speechiness", "instrumentalness", "liveness", "duration_ms"]

# Columnas de metadatos que se conservan aparte (NUNCA entran al modelo).
ID_COL = "track_id"
GENRE_COL = "track_genre"
META_COLS = ["track_id", "track_name", "artists", "album_name", "popularity", "explicit"]

# --- Limpieza ----------------------------------------------------------------
# Una canción puede aparecer en varios géneros: se conserva la primera aparición
# como "género principal" y se guarda cuántos géneros tenía (n_genres).
DEDUP_BY_NAME = True  # además de track_id, quita mismo (artista, nombre) repetido

# --- PCA ---------------------------------------------------------------------
PCA_VARIANCE_TARGET = 0.90  # varianza acumulada mínima para elegir nº de componentes

# --- Muestra para clustering jerárquico --------------------------------------
# El jerárquico aglomerativo es O(n^2) en memoria: con >100k canciones no corre.
# Se entrega una muestra estratificada por género marcada con la columna in_sample.
SAMPLE_SIZE = 20_000
