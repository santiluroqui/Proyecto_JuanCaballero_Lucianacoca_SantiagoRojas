# 🎵 Perfiles musicales mediante aprendizaje no supervisado

Proyecto del módulo **Aprendizaje No Supervisado** (Universidad Católica Boliviana). Equipo: Luciana · Santiago · Juan.

**Pregunta de investigación.** ¿Es posible descubrir agrupaciones musicales coherentes usando únicamente características acústicas medibles, y qué tan bien se corresponden con los géneros tradicionales?

## Idea en una imagen

```
Canción ──► 10 variables acústicas ──► limpieza + log + estandarización ──► PCA ──► K-Means / Jerárquico
                                                                                          │
                      género real (guardado aparte, nunca usado para entrenar) ◄── comparación a posteriori
```

El género **no** es una entrada del modelo: se usa solo al final para contrastar los clusters descubiertos contra la clasificación humana.

## Estado del proyecto

| Módulo | Contenido | Responsable | Estado |
|---|---|---|---|
| 0 | Setup y exploración inicial | Santiago | ✅ código listo |
| 1 | EDA | Santiago · Luciana | ✅ código listo |
| 2 | Preprocesamiento | Santiago | ✅ código listo |
| 3 | PCA | Santiago | ✅ código listo |
| 4 | Clustering | Juan | ⏳ |
| 5 | Evaluación de modelos | Juan | ⏳ |
| 6 | Interpretación | Juan · Luciana | ⏳ |
| 7 | Recomendador (opcional) | Juan · Santiago | ⏳ |
| 8 | Dashboard interactivo | Santiago · Juan | ⏳ |

> Los resultados numéricos y gráficos se publicarán aquí cuando el pipeline corra sobre el dataset real.

## Cómo reproducirlo

```bash
git clone <url-del-repo> && cd proyecto-perfiles-musicales
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Descargar dataset.csv a data/raw/  (instrucciones en data/raw/README.md)
# 2) Preprocesamiento + PCA en un comando
python scripts/run_pipeline.py
# 3) Explorar paso a paso
jupyter lab notebooks/
```

El pipeline es determinista (`random_state = 42`).

## Datos

[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (Pandya, 2023): ~114.000 pistas, 125 géneros, audio features. Se usa el CSV público; no se consulta la API de Spotify.

## Estructura

```
├── notebooks/     00 exploración · 01 EDA · 02 preprocesamiento · 03 PCA · (04-07 por completar)
├── src/           config.py (decisiones metodológicas) · preprocessing.py · viz.py
├── scripts/       run_pipeline.py (reproduce módulos 2-3)
├── data/          raw/ (no versionado) · processed/ (generado)
├── reports/       figures/ · tables/
├── dashboard/     app interactiva (módulo 8)
├── docs/          informe y presentación
├── HANDOFF_JUAN.md · CONTRIBUTIONS.md
```

## Decisiones metodológicas clave

Todas centralizadas en [`src/config.py`](src/config.py) para poder discutirlas y cambiarlas en un solo lugar:
variables usadas, variables con `log1p`, tamaño de la muestra para el jerárquico, umbral de varianza de PCA y semilla.

## Créditos

Luciana (informes y documentación) · Santiago (datos, preprocesamiento, PCA, dashboard) · Juan (clustering, evaluación, interpretación).
