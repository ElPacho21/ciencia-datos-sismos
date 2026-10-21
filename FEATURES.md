# Features y objetivos del dataset de réplicas

Qué columnas se pueden usar para entrenar sin fuga, desde cuándo se conocen y cuáles
tienen nulos estructurales. "Instante de predicción" = `mainshock_time`, salvo en
`horizons_*.csv`, donde es `mainshock_time + horizon_hours`.

## `clusters_*.csv` (una fila por secuencia)

| Columna | Qué mide | Disponible desde | ¿Existiría al predecir? | Nulos estructurales |
|---|---|---|---|---|
| `mainshock_mag` | Magnitud del sismo principal | t = 0 | Sí* | No |
| `mainshock_depth` | Profundidad | t = 0 | Sí | No |
| `mainshock_lat` / `mainshock_lon` | Epicentro | t = 0 | Sí | No |
| `mainshock_time` | Instante del principal | t = 0 | Sí (sólo para ordenar o separar por tiempo) | No |
| `n_prior_2d` / `n_prior_30d` / `n_prior_365d` | Eventos ≥ Mc a < `prior_radius_km` en los 2 / 30 / 365 días previos | t = 0 | Sí | No (0 es un valor) |
| `prior_max_mag_365d` | Mayor magnitud de esos eventos del último año | t = 0 | Sí | Sí: sin eventos previos |
| `days_since_prior` | Días desde el evento previo más reciente | t = 0 | Sí | Sí: ninguno desde `starttime` |
| `prior_window_truncated` | La ventana de 365 d empieza antes que el catálogo | t = 0 | Sí | No |
| `region_mc` / `region_log10_eta0` | Mc y umbral de la región | t = 0 | Sí (parámetros del método) | No; constantes dentro de una región |
| `dist_to_border_km` / `near_border` | Cercanía al borde del rectángulo | t = 0 | Sí | No |
| `region` | Región (sólo en el combinado) | t = 0 | Sí | No |
| `n_aftershocks` | Réplicas de toda la secuencia | **posterior** | **No: es el objetivo** | No |
| `has_aftershock` | `n_aftershocks > 0` | **posterior** | **No: es el objetivo** | No |
| `n_events`, `n_foreshocks`, `duration_days`, `extent_km`, `max_generation`, `root_is_mainshock`, `cluster_id` en su rol de conteo | Describen la secuencia ya terminada | **posterior** | **No: fuga** | No |

\* Ver "Advertencias de causalidad".

## `horizons_*.csv` (una fila por cluster y horizonte)

| Columna | Qué mide | Disponible desde | ¿Existiría al predecir? | Nulos estructurales |
|---|---|---|---|---|
| `horizon_hours` | Instante de predicción `t` | t | Sí | No |
| `n_aft_t` | Réplicas hasta `t` | t | Sí* | No |
| `max_aft_mag_t` | Mayor magnitud entre ellas | t | Sí* | Sí: ninguna réplica hasta `t` |
| `hours_since_last_t` | Horas desde el último evento visto | t | Sí* | No (cuenta el principal) |
| `y_aft`, `y_has_aft` | **Objetivo**: réplicas entre `t` y `T` días | posterior | No: es el objetivo | No |
| `n_aft_T` | Réplicas hasta `T` | posterior | No: fuga | No |
| `censored_T`, `censored_t` | El catálogo no cubre la ventana | — | Metadato para filtrar | No |

## Advertencias de causalidad

- **Sismo principal = el mayor del cluster** (`mainshock=largest`). Se sabe recién cuando
  ocurrió el mayor, y a veces después hubo premonitores más chicos. `mainshock_mag` es
  causal en el instante del principal, pero la *selección* de qué evento es el principal
  usa información posterior. Con `mainshock=root` esto no pasa.
- **Pertenencia al cluster y η₀** se ajustan con el catálogo completo: las features de
  `horizons_*.csv` ("réplicas hasta `t`") no se podrían recomputar con sólo lo ocurrido
  hasta `t`.
- **Separar train y test por `cluster_id`** (`GroupKFold`): las filas de un mismo cluster
  en `horizons_*.csv` comparten secuencia.
- **Regiones:** Mc y η₀ cambian por región. Con `region` como feature o `GroupKFold`
  por región se puede medir si el modelo generaliza entre regiones.
- **Efecto borde:** las secuencias `near_border` pudieron perder réplicas afuera del
  rectángulo; se pueden excluir o ponderar.
- **Tope de distancia:** sin `r_max_mode=gk` hay clusters de más de 1.500 km
  (encadenamiento de sismicidad no relacionada) y `n_aftershocks` queda inflado.

## Población de las corridas por región

Corrida con los defaults (sin tope de distancia), ventanas de `regions.yaml`:

| Región | Secuencias | % sin réplicas | Con ≥1 réplica | M>6 | % censuradas (T=30 d) | % `near_border` |
|---|---|---|---|---|---|---|
| andes | 3.897 | 91,6 | 326 | 88 | 0,3 | 5,6 |
| japan | 3.216 | 88,9 | 356 | 87 | 0,4 | 13,0 |
| philippines | 1.998 | 86,8 | 263 | 60 | 0,3 | 21,9 |
| sumatra_java | 3.211 | 88,4 | 372 | 74 | 0,3 | 9,2 |
| mexico_ca | 7.248 | 89,6 | 752 | 92 | 0,3 | 5,3 |
| **Total** | **19.570** | **89,4** | **2.069** | **401** | **0,3** | **9,0** |

Con `r_max_mode=gk`, `r_max_k=3`: 21.292 secuencias, 2.065 con réplicas, 448 con M>6 y
la extensión máxima de un cluster baja de 3.732 km a 533 km. La separación de modos de la
mezcla queda en 3,15 (andes), 2,62 (japan), 3,16 (philippines), 2,62 (sumatra_java) y
1,53 (mexico_ca); sin tope era 2,58 / 2,45 / 2,77 / 2,46 / 2,13.

## Qué tanto predice cada feature (sin censuradas, 19.505 filas)

`scripts/feature_correlations.py`; la brecha es la media en secuencias con réplicas menos la
media en las que no, en desvíos de la feature.

| Feature | Pearson | Spearman | Brecha |
|---|---|---|---|
| `mainshock_mag` | 0,171 | 0,340 | 1,498 |
| `mainshock_depth` | −0,011 | −0,136 | −0,242 |
| `n_prior_2d` | 0,149 | 0,127 | 0,218 |
| `n_prior_30d` | 0,063 | 0,065 | 0,174 |
| `n_prior_365d` | 0,005 | 0,039 | 0,147 |
| `prior_max_mag_365d` | 0,014 | 0,035 | 0,125 |
| `days_since_prior` | −0,004 | −0,086 | −0,067 |
| `dist_to_border_km` | −0,001 | −0,025 | −0,085 |

La magnitud del principal domina; las features de sismicidad previa aportan una señal
chica (la de 2 días, la mejor). Es un resultado honesto: el modelo del que se parte es
"magnitud + algo de contexto", y no hay que esperar milagros. En Japón `n_prior_2d` tiene
Pearson 0,634 con Spearman 0,163: lo arrastran pocas secuencias (el enjambre de Tohoku),
no una relación general.

## Qué agrega al modelo

`scripts/feature_baseline.py` mide el aporte con un `HistGradientBoosting` sobre
`log1p(n_aftershocks)`, contra dos predicciones triviales. Con 19.570 secuencias:

| modelo | R² | MAE(log) | MAE réplicas |
|---|---|---|---|
| trivial: 0 réplicas (la mediana) | −0,077 | 0,115 | 0,834 |
| trivial: media de log1p | 0,000 | 0,206 | 0,930 |
| magnitud + profundidad + posición + región | 0,559 | 0,126 | 0,784 |
| + sismicidad previa | 0,575 | 0,124 | 0,779 |

Con corte temporal 70/30 (viejas a train, nuevas a test) el salto es 0,520 → 0,554. Las
features nuevas suman entre 0,02 y 0,03 de R²; `n_prior_2d`, `n_prior_365d` y
`days_since_prior` son las que aportan. Nada de eso mueve el MAE en réplicas, que está
dominado por los ceros y por cuatro secuencias gigantes.

El valor de `n_prior_2d` se ve mejor condicionando por magnitud. En M5-6, la proporción
de secuencias con al menos una réplica pasa de 21,0 % (sin actividad previa) a 43,6 %
(1-4 eventos) y 81,2 % (5 o más); en M6-7, de 75,1 % a 92,2 % y 100 %. No está confundida
con la magnitud (Spearman entre ambas: 0,009).

Dos advertencias para cuando se entrene en serio: el 89,4 % del objetivo es cero, así que
conviene `log1p`, Poisson o un modelo en dos etapas (clasificar `has_aftershock` y después
contar); y el bloque `n_prior_*` cuenta también los premonitores de la misma secuencia, o
sea que parte de su señal es "esta secuencia ya venía grande". Para un escenario en tiempo
real conviene la tabla de horizontes, donde lo observado hasta `t` es causal de verdad:
a 1 h, el R² de `y_aft` pasa de 0,495 a 0,552 al sumar `n_aft_t`, `max_aft_mag_t` y
`hours_since_last_t`.

## Corrida de referencia (defaults)

Con los parámetros por defecto la corrida original no cambió: 2.321 eventos ≥ Mc, 1.978
secuencias, 131 con réplicas, Σ`n_aftershocks` = 279, y `n_aftershocks` idéntico al csv
de referencia. Las columnas nuevas van al final.
