# Entrega de Santiago → Juan (Módulos 0–3)

Juan: aquí está todo lo necesario para empezar el **Módulo 4 (clustering)** sin tener que tocar el preprocesamiento.

## 1. Cómo generar los archivos

```bash
pip install -r requirements.txt
# 1) colocar dataset.csv en data/raw/  (ver data/raw/README.md)
python scripts/run_pipeline.py          # deja todo en data/processed/
```

Es determinista (`random_state = 42`): tú y yo obtenemos exactamente los mismos archivos.
También se puede ejecutar paso a paso con los notebooks `02_preprocesamiento` y `03_pca`.

## 2. Qué archivos recibes (`data/processed/`)

| Archivo | Contenido | Para qué lo usas |
|---|---|---|
| `X_scaled.csv` | `track_id` + 10 variables transformadas y estandarizadas (media 0, desv. 1) | Clustering en el espacio original |
| `X_pca.csv` | `track_id` + componentes principales (las necesarias para ≥ 90 % de varianza) | Clustering en espacio reducido y visualización |
| `songs_clean.csv` | `track_id`, metadatos, `primary_genre`, `n_genres`, variables **originales** (sin escalar), `in_sample` | Interpretar clusters (centroides en unidades reales) y comparar con género |
| `scaler.joblib`, `pca.joblib` | Objetos ajustados | Proyectar canciones nuevas (recomendador, dashboard) |
| `config_preprocesamiento.json` | Variables usadas, nº de filas, nº de componentes, varianza acumulada | Citar cifras exactas en el informe |

Las tablas de apoyo están en `reports/tables/` (`bitacora_limpieza.csv`, `pca_varianza_explicada.csv`, `pca_loadings.csv`) y los gráficos en `reports/figures/`.

**Las tres tablas tienen el mismo orden de filas** y comparten `track_id`; `pp.load_preprocessed()` verifica la alineación.

## 3. Cómo cargarlos

```python
import sys; sys.path.insert(0, "..")          # si trabajas desde notebooks/
import pandas as pd
from src import config as cfg

songs = pd.read_csv(cfg.DATA_PROCESSED / "songs_clean.csv")
X_pca = pd.read_csv(cfg.DATA_PROCESSED / "X_pca.csv").drop(columns="track_id")
X_sca = pd.read_csv(cfg.DATA_PROCESSED / "X_scaled.csv").drop(columns="track_id")

# Muestra estratificada para el jerárquico (O(n²) en memoria):
mask = songs["in_sample"].values
X_pca_sample = X_pca[mask]
```

## 4. Reglas del juego (importante)

1. **El género nunca entra al modelo.** Ya está fuera de `X_scaled.csv` y `X_pca.csv`; vive solo en `songs_clean.csv` (`primary_genre`) para el Módulo 6.
2. **K-Means** puede correr con todas las filas. **El jerárquico aglomerativo no:** usar `in_sample == True` (≈ 20.000 canciones, proporcional por género). Para comparar ambos métodos de forma justa, evalúen K-Means también sobre esa misma muestra.
3. **Multigénero:** una canción listada en varios géneros se conservó una sola vez (primer género = `primary_genre`; `n_genres` guarda cuántos tenía). Al hacer la matriz de contingencia, tengan presente que el género de referencia es "el primero listado", no "el verdadero".
4. **Dos espacios posibles** para clusterizar: `X_scaled` (10 dimensiones) o `X_pca` (k dimensiones). Sugerencia: correr ambos y comparar silhouette; si dan resultados parecidos, usar PCA (más rápido y se visualiza igual).
5. **Interpretar siempre en unidades originales:** calcular centroides con las variables de `songs_clean.csv` agrupadas por cluster, no con las estandarizadas.

## 5. Decisiones de preprocesamiento (para citar en el informe)

- **Variables (10):** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms.
- **Limpieza:** nulos; `tempo = 0` y `duration_ms = 0` (pistas vacías); duplicados por `track_id` (multigénero) y por artista + nombre. Cada paso queda contado en `reports/tables/bitacora_limpieza.csv`.
- **Transformación:** `log1p` en speechiness, instrumentalness, liveness y duration_ms (lista editable en `src/config.py → LOG_FEATURES`). Se confirma con el notebook 01 que reduce la asimetría; si alguna no mejora, se quita de la lista y se vuelve a correr el pipeline.
- **Outliers:** se reportan (regla IQR) pero **no se eliminan**; en música los valores extremos suelen ser reales.
- **Estandarización:** `StandardScaler`.
- **PCA:** se conserva el menor nº de componentes con ≥ 90 % de varianza acumulada (`PCA_VARIANCE_TARGET`).

## 6. Estado y pendientes

- [x] Código del pipeline y de los 4 notebooks probado de punta a punta con datos **sintéticos** de igual esquema (no se pudo descargar el dataset real desde el entorno donde se escribió).
- [ ] **Correr con el dataset real** y completar las celdas "Hallazgos / Conclusiones" de los notebooks con números reales.
- [ ] Confirmar con el EDA real si `LOG_FEATURES` es adecuada.
- [ ] Decidir juntos: ¿todos los géneros o un subconjunto? (ver notebook 00, sección 6).
- [ ] Juan: avisar si necesita otra transformación o más componentes.
