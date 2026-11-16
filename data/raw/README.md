# Datos crudos

Esta carpeta debe contener **`dataset.csv`** (no se sube a Git por tamaño).

## Fuente

**Spotify Tracks Dataset** (Pandya, M., 2023) — ~114.000 pistas de 125 géneros con sus *audio features*, formato CSV, licencia BSD.

| Opción | Enlace |
|---|---|
| Hugging Face (recomendada) | https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset |
| Kaggle | https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset |

## Cómo obtenerlo

1. Entra al enlace de Hugging Face → pestaña **Files and versions** → descarga `dataset.csv`.
2. Guárdalo exactamente como `data/raw/dataset.csv`.

Alternativa por código (con internet abierto):

```python
import pandas as pd
df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
df.to_csv("data/raw/dataset.csv", index=False)
```

## Nota metodológica

Los datos fueron originalmente extraídos con la API de Spotify, pero este proyecto **no consulta la API**: usa el CSV público ya publicado.
En el informe, citar la fuente en APA y mencionar ese origen.

## Esquema esperado (21 columnas)

`Unnamed: 0, track_id, artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, track_genre`
