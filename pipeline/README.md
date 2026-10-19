# Pipeline — detector de réplicas sísmicas

Proyecto de Astronomer (Astro Runtime 3.3-4, Airflow 3). Es autocontenido: todo
lo que hace falta para levantarlo está en esta carpeta.

## Levantarlo

```bash
cd pipeline
astro dev start
```

La interfaz queda en http://localhost:8080. La conexión `usgs_fdsn` se crea
sola desde `airflow_settings.yaml`, así que el DAG se puede disparar sin tocar
nada más.

`detector_replicas_api` no tiene schedule: se dispara a mano con *Trigger DAG
w/ config*. Los parámetros de la consulta son `starttime`, `endtime`,
`minmagnitude`, el rectángulo `minlatitude`/`maxlatitude`/`minlongitude`/`maxlongitude`
(por defecto Argentina continental; los cuatro en null consultan el planeta) y
`limit` (un tope de seguridad: si el rango empareja más sismos que eso la corrida
falla en vez de truncar; `0` es sin tope). Del método salen `mainshock`
(`largest`/`root`), `seed` y `n_randomizations`.
`r_max_km` (por defecto `null`, sin tope) fija la distancia epicentral máxima
entre padre e hijo: un par más lejos no puede ser padre-hijo, y un evento sin
candidato dentro del radio queda como raíz (`eta` en NaN). `r_max_mode="gk"`
usa un radio que depende de la magnitud del padre, `r_max_k · 10^(0.1238·M + 0.983)` km
(Gardner-Knopoff), y `r_max_km` pasa a ser un techo absoluto opcional. El mismo
tope se aplica al catálogo nulo. Si está activo, aparece en el nombre de los
archivos desde `neighbors/` en adelante (`_rmax=300`, `_rmax=gk3`); sin tope los
nombres son los de siempre.
`prior_radius_km` (100 por defecto) es el radio de la vecindad con la que
`build_features` cuenta la sismicidad previa; si se cambia, el csv publicado lleva
`_prior=<km>` en el nombre. Y `force` ignora todo lo ya
persistido y recalcula de cero.

La ventana tiene que ser **larga**. El método necesita al menos 50 eventos por
encima de la magnitud de completitud, y `estimate_mc` corta la corrida si no
llegan. Con cinco semanas de Argentina quedan ocho y falla; con diez años,
sobran:

```
starttime = 2016-01-01   endtime = 2026-01-01   minmagnitude = 3.5
```

Los csv de salida quedan en `include/output/detector_replicas_api/delivery/`.

## Estructura

| Ruta | Qué hay |
|---|---|
| `dags/` | Orquestación: schedule, params, dependencias entre tareas. |
| `include/sismos/` | La lógica importable. Se llega como `import sismos` gracias al `PYTHONPATH` del Dockerfile. |
| `include/output/` | Salida de cada paso: `bronze/` y `silver/` con el catálogo, `neighbors/`, `threshold/`, `null/`, `aftershocks/` y `clusters/` con lo que produce el método, y **`delivery/` con los csv finales**. Lo que depende de la semilla la lleva en el nombre. No se versiona: se regenera corriendo el DAG. |
| `tests/dags/` | Chequeos de integridad del DAG. |
| `airflow_settings.yaml` | Conexiones del entorno local. Hoy no tiene secretos. |

La lógica va en `include/sismos/` y no en `dags/` por dos razones: se puede
testear sin levantar Airflow, y los notebooks de la raíz del repo pueden
importar exactamente el mismo código que corre en producción.

Ojo con el contexto de build: `astro deploy` empaqueta **esta** carpeta, así
que nada que esté por encima de `pipeline/` entra en la imagen.

## Tests

```bash
cd pipeline
astro dev pytest
```
