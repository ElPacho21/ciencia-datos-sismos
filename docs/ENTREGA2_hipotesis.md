# Entrega 2 — Análisis exploratorio (EDA): qué falta y hipótesis propuestas

> Contexto para vos, Claude Code: esto es la **Entrega 2** del proyecto integrador
> (detector de réplicas sísmicas), miércoles 23/09. Es la continuación de la Entrega 1
> que ya está resuelta (commit `2a72f9d`, ver `EXPLICACION_PROYECTO.md` en el repo).
> Tenés más contexto del código real y de las columnas efectivas del dataset que yo,
> así que este documento te da lo que el humano ya definió y te pide que vos
> completes/corrijas el resto con ese contexto.

## 1. Qué pide la cátedra (resumen operativo)

La Entrega 2 es una defensa oral de 20 minutos, individual dentro del grupo. Hay que
llegar con:

1. **Notebook de EDA corrido de punta a punta**, sin errores, con salidas guardadas.
2. **Perfil del dataset** con números concretos: `shape`, `dtypes.value_counts()`,
   nulos por columna (`isna().mean()`), columnas constantes (`nunique() <= 1`),
   duplicados de la clave, distribución de la variable objetivo, asimetría de las
   numéricas (`skew()`).
3. **Cuatro fichas de hipótesis** (ver formato abajo).
4. **Tabla de columnas candidatas** para el modelo, con chequeo de fuga por columna.
5. Los puntos que la Entrega 1 dejó pendientes, resueltos o con explicación de por
   qué no se pudieron resolver (ver `EXPLICACION_PROYECTO.md`, sección 7 — ya están
   todos resueltos salvo el 7.6, que es justamente uno de los temas de esta entrega).

**El número más importante de toda la entrega: la distribución de la variable
objetivo.** Ya sabemos que `n_replicas` está inflado de ceros (93%, ver punto 7.6 de
`EXPLICACION_PROYECTO.md`) — eso hay que mostrarlo con el número, no como algo que se
menciona de pasada.

### Formato exacto de cada ficha de hipótesis (6 campos obligatorios)

Tomado del material de cátedra (`Hipotesis_sobre_el_canonico.md`, que trabaja un
ejemplo con el dataset FIFA y dice explícitamente que el mismo método se traslada tal
cual al proyecto propio):

| Campo | Qué va |
|---|---|
| **Hipótesis** | La afirmación, escrita **antes** de correr la medida. No vale redactarla mirando el resultado. |
| **Pregunta** | `responder` (pregunta de dominio/negocio → termina en una conclusión) o `predecir` (pregunta de modelado → termina en una decisión sobre qué columnas usar). |
| **Plantilla** | Una de las tres: **comparación** (·X difiere entre A y B· → separación estandarizada, o η² con >2 grupos), **asociación** (·X se mueve junto con Y· → Pearson y Spearman), **composición** (·C está sobrerrepresentado en G· → proporción del grupo contra la global). La plantilla determina la medida; no se elige la medida después de ver cuál conviene. |
| **Medida** | El número que sale de calcular lo anterior. |
| **Zona** | Semáforo: **verde** (confirma/refuta con claridad), **rojo** (refutada), **amarillo** (ambigua — habilita hasta dos movimientos: partir la población, controlar una tercera variable, cambiar la escala; después de eso se escribe *inconclusa*, no se sigue iterando). |
| **Decisión** | Qué se hace con el resultado: entra al modelo, sale por fuga, se transforma, queda inconclusa para que decida el modelo de la Unidad 3, etc. |

### Reglas duras sobre las 4 hipótesis (no negociables)

- Al menos una debe ser **"responder"** (dominio) y al menos una **"predecir"**
  (decide sobre una columna del modelo).
- Al menos una debe terminar **refutada o inconclusa**. Cuatro verdes/confirmadas es
  señal de que se eligieron hipótesis "seguras" que ya se sabían de antemano.
- Cada resultado se sostiene con el **número**, no con el gráfico. El gráfico
  explica el número, no lo reemplaza ("se ve que hay diferencia" no es un resultado).
- Ninguna hipótesis debe pasar por **más de dos movimientos** en zona amarilla antes
  de declararse inconclusa.

### La tabla de columnas candidatas

Una fila por columna que se le vaya a dar al modelo (de `clusters_*.csv` y/o
`eventos_*.csv`), con: qué mide, zona, decisión (entra / sale / entra transformada),
y la columna de chequeo de fuga: **¿existiría esa columna en el momento de predecir?**
(ej.: `n_eventos = n_replicas + n_premonitores + 1` casi seguro es fuga si se quiere
predecir `n_replicas`, porque conocer `n_eventos` implica ya conocer casi la respuesta).

## 2. Las columnas reales (de `EXPLICACION_PROYECTO.md`)

**`clusters_*.csv`** (entregable principal — 1.978 filas × 14 columnas, clave
`cluster_id`, objetivo `n_replicas`): `cluster_id`, `mainshock_id`, `mainshock_mag`,
`mainshock_time`, `lat`, `lon`, `n_eventos`, `n_replicas`, `n_premonitores`,
`duracion_dias`, `extension_km`, `generacion_max`, `raiz_id`, `raiz_es_mainshock`.

**`eventos_*.csv`** (2.321 filas, clave `id`, objetivo `is_aftershock`): `id`,
`cluster_id`, `generacion`, `orden_en_cluster`, `is_mainshock`, `is_aftershock`,
`is_aftershock_eta` (el corte duro, antes del thinning probabilístico), más los
campos crudos del catálogo (magnitud, lat, lon, tiempo) y las columnas de la métrica
η: `log10_T`, `log10_R`, `log10_eta`.

**Ojo:** si estas columnas cambiaron desde que se escribió `EXPLICACION_PROYECTO.md`
(por ejemplo si agregaste/sacaste algo en un commit posterior), tomá como fuente de
verdad el código y el CSV real, no este listado.

## 3. Hipótesis que propongo (punto de partida, no cerradas)

Las pensé mirando el documento del proyecto, **sin correr el notebook** — o sea que
no verifiqué los números reales, solo que la pregunta tenga sentido con lo que sé de
las columnas. Puede que al calcularlas alguna no dé lo que espero, o que directamente
no sea la más interesante que se puede sacar del dataset. Vos tenés el código y podés
correr todo, así que:

**Quiero que las uses como punto de partida, pero que las evalúes con criterio
propio y las reemplaces o ajustes las que no aporten tanto como otras que se te
ocurran mirando el dataset real.** No hace falta que sean estas 4 exactas — el
objetivo es llegar a las 4 mejores hipótesis posibles cumpliendo las reglas de la
sección 1, no defender estas en particular.

---

**H1 — Composición · "responder" (dominio).**
*La discrepancia entre mainshock "mayor" y mainshock "raíz" no es uniforme: se
concentra en clusters con premonitores grandes.*
Ya se sabe que 42 clusters (2,1%) tienen `raiz_es_mainshock = False`. La hipótesis es
que esos casos no están dispersos al azar, sino que se concentran donde hubo un
premonitor de magnitud cercana a la del mainshock real (`n_premonitores > 0` y una
diferencia de magnitud chica). Medida: proporción de esos 42 clusters que cumple esa
condición, contra la proporción esperada si fuera aleatorio.
Zona esperada: verde (caracteriza algo que ya se sabe que existe).

**H2 — Asociación · "predecir" (ley de Utsu, columna candidata).**
*`n_replicas` crece con `mainshock_mag`, pero no de forma lineal.*
Pearson vs Spearman entre `mainshock_mag` y `n_replicas` (probablemente conviene
`log1p(n_replicas)` dado el inflado de ceros). Con el 93% de ceros en el target, es
candidata fuerte a salir amarilla/inconclusa — lo cual está bien, la consigna pide
al menos una así.

**H3 — Comparación · "predecir" (chequeo de fuga).**
*`n_eventos` predice `n_replicas` casi perfectamente porque son la misma cantidad
disfrazada* (`n_eventos = n_replicas + n_premonitores + 1`, identidad ya documentada
en el proyecto).
Esto es candidato directo a la tabla de columnas: si se confirma la relación casi
determinística, la decisión es "sale por fuga" — `n_eventos` no existiría en el
momento de predecir `n_replicas`.

**H4 — Comparación · "responder" (calidad del método, conecta con el punto 7.6
abierto).**
*El thinning probabilístico (`is_aftershock`) difiere del corte duro
(`is_aftershock_eta`) más en la zona intermedia de η que en los extremos.*
Separación estandarizada de `log10_eta` entre eventos donde ambas columnas coinciden
vs donde discrepan. Conecta con el "exceso de 6 puntos contra el catálogo barajado"
que ya está documentado, y le da contexto cuantitativo al problema de inflación de
ceros que hay que mencionar en la entrega.

---

## 4. Si armás hipótesis propias en vez de (o adicionales a) estas

Recordá que tienen que cumplir las reglas duras de la sección 1: mezcla de
"responder"/"predecir", al menos una no-verde, medida antes que gráfico, y máximo
dos movimientos en amarillo. Si el dataset real da resultados distintos a los que
supuse arriba (por ejemplo si H2 sale verde clarísimo, o H1 no se sostiene), está
bien — lo que importa es que la ficha final refleje lo que realmente dio el cálculo,
no lo que yo anticipé acá.

## 5. Qué necesito que hagas

1. Revisá si el notebook de EDA para este proyecto (terremotos) ya existe en el repo
   o si hay que armarlo desde cero — a diferencia del dataset FIFA de la materia, acá
   no tenemos un EDA previo hecho.
2. Calculá el perfil del dataset (sección 1, punto 2) sobre `clusters_*.csv` y
   `eventos_*.csv` reales.
3. Armá las 4 fichas de hipótesis: tomá las mías como base, ajustalas o reemplazalas
   con tu criterio y los datos reales, y dejalas con los 6 campos completos y el
   número real (no estimado).
4. Armá la tabla de columnas candidatas para el modelo de la Unidad 3, con el
   chequeo de fuga por columna.
5. Revisá el punto 7.6 de `EXPLICACION_PROYECTO.md` (target inflado de ceros) — es
   territorio compartido entre "lo que quedó abierto de la Entrega 1" y "lo que hay
   que mostrar en el perfil del dataset" de esta entrega.
