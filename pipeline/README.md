# Pipeline — detector de réplicas sísmicas

Proyecto de Astronomer (Astro Runtime 3.3-4, Airflow 3). Es autocontenido: todo
lo que hace falta para levantarlo está en esta carpeta.

## Levantarlo

```bash
cd pipeline
astro dev start
```

La interfaz queda en http://localhost:8080. La conexión `sismosapi` se crea
sola desde `airflow_settings.yaml`, así que el DAG se puede disparar sin tocar
nada más.

`detector_replicas_api` no tiene schedule: se dispara a mano con *Trigger DAG
w/ config*. Los parámetros de la consulta son `starttime`, `endtime`,
`minmagnitude`, el rectángulo `minlatitude`/`maxlatitude`/`minlongitude`/`maxlongitude`
(por defecto Argentina continental; los cuatro en null consultan el planeta) y
`limit` (un tope de seguridad: si el rango empareja más sismos que eso la corrida
falla en vez de truncar; `0` es sin tope). Del método salen `mainshock`,
`seed` y `n_randomizaciones`. Y `force` ignora todo lo ya persistido
y recalcula de cero.

## Estructura

| Ruta | Qué hay |
|---|---|
| `dags/` | Orquestación: schedule, params, dependencias entre tareas. |
| `include/sismos/` | La lógica importable. Se llega como `import sismos` gracias al `PYTHONPATH` del Dockerfile. |
| `include/output/` | Salida de cada paso: `bronze/` y `silver/` con el catálogo, y `vecinos/`, `umbral/`, `nulo/`, `replicas/` y `clusters/` con lo que produce el método. Lo que depende de la semilla la lleva en el nombre. No se versiona: se regenera corriendo el DAG. |
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
