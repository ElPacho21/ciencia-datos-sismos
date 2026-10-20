# Detector de réplicas sísmicas — explicación completa del proyecto

> Documento de estudio para la Entrega 1. Cubre el **dominio** (sismología) y lo
> **técnico** (qué se implementó y por qué). Complementa al `README.md` de la
> raíz, que está escrito para responder "qué hace cada paso y por qué está ahí".
>
> **Actualizado al commit `2a72f9d` ("feat: publicar el dataset en csv y
> validarlo").** Ese commit cerró tres cosas que este documento marcaba como
> pendientes: el `NameError` que rompía el DAG, la salida en CSV, y la bisección
> del rango de fechas. Ver la sección 7.

**Índice**

1. [El dominio: qué problema sísmico se resuelve](#1-el-dominio-qué-problema-sísmico-se-está-resolviendo)
2. [El método Zaliapin & Ben-Zion (el corazón científico)](#2-el-método-zaliapin--ben-zion-el-corazón-científico)
3. [La arquitectura técnica](#3-la-arquitectura-técnica)
4. [Tarea por tarea del DAG](#4-tarea-por-tarea-del-dag)
5. [El dataset de salida y los 7 criterios de ENTREGA1](#5-el-dataset-de-salida-y-los-7-criterios-de-entrega1)
6. [Cómo se valida que las cuentas están bien](#6-cómo-se-valida-que-las-cuentas-están-bien)
7. [Puntos abiertos y su estado](#7-puntos-abiertos-y-su-estado)
8. [Cómo mapear esto a las preguntas del oral](#8-cómo-mapear-esto-a-las-preguntas-del-oral)

---

## 1. El dominio: qué problema sísmico se está resolviendo

### 1.1. Qué es un catálogo sísmico y de dónde sale

Un **catálogo sísmico** es una lista plana de terremotos. Cada fila: fecha/hora,
latitud, longitud, profundidad, magnitud, y metadatos (qué red lo midió, tipo de
magnitud, errores de localización). Nada más.

La fuente acá es el **servicio FDSN de eventos del USGS**
(`earthquake.usgs.gov/fdsnws/event/1/`). FDSN es un estándar internacional de
webservices sismológicos; el endpoint `query` devuelve eventos en
CSV/GeoJSON/QuakeML y el endpoint `count` te dice cuántos hay sin bajarlos. Es
abierto, sin API key.

Dato relevante para Argentina: el USGS es una **red global**. Ve bien los sismos
M≥4.5 del país pero se le escapan casi todos los más chicos, que los tiene el
**INPRES** (la red nacional). Esto reaparece cuando hablemos de Mc.

### 1.2. Los terremotos no son independientes

Después de un sismo grande vienen **cientos de réplicas** (aftershocks): sismos
más chicos apretados en el tiempo (días/semanas) y en el espacio (decenas a
cientos de km) alrededor del primero. A veces también hay **premonitores**
(foreshocks) antes del grande. Al sismo dominante de la secuencia se lo llama
**sismo principal** (mainshock).

Tres regularidades empíricas que conviene poder nombrar en el oral:

| Ley | Qué dice | Dónde aparece en el código |
|---|---|---|
| **Gutenberg-Richter** | El número de sismos decae exponencialmente con la magnitud: `log₁₀ N(≥M) = a − b·M`. El **b-value** ≈ 1.0 en casi todo el mundo (hay ~10× más sismos por cada unidad de magnitud menos). | `parameters.py` → `fmd()`, y `b=1.0` estándar |
| **Omori(-Utsu)** | La tasa de réplicas decae con el tiempo como `~1/tᵖ` después del principal. Consecuencia: las réplicas también tienen réplicas → **cadenas largas**. | Motiva `build_clusters` (hay que seguir la cadena, no agrupar por padre directo) |
| **Productividad de Utsu** | El número medio de réplicas escala como `~10^(α·M)` con la magnitud del principal. `α` ≈ 0.8–1.0 en el mundo. | `clusters.py` → `productivity()`, es el control externo del pipeline |
| **Dimensión fractal de epicentros (`d`)** | Los epicentros no llenan el plano uniformemente: se acumulan sobre fallas. `d ≈ 1.5` mide ese "amontonamiento". | `d=1.5` estándar en `parameters.py` |

### 1.3. Declustering: por qué esto es un problema que vale la pena

**Declustering** = separar los sismos "de fondo" (independientes, tectónicos) de
los "disparados" (réplicas y premonitores).

Es el paso previo a casi toda la estadística sísmica. Si preguntás *"¿cuántos
sismos por año hay en esta zona?"* sin declusterizar, la respuesta está inflada:
una sola secuencia de réplicas mete miles de eventos que en realidad son **un
evento con muchas colas**. Lo mismo para estimación de peligro sísmico, cálculo
de tasas, mapas de amenaza, etc.

### 1.4. Los dos enfoques de declustering

- **Clásico (Gardner–Knopoff, 1974):** ventanas espacio-temporales fijas. *"Todo
  lo que ocurra dentro de 50 km y 10 días de un M6 es réplica"*. El problema: esas
  ventanas son **tablas que alguien decidió una vez** y no se adaptan a la región
  ni al catálogo.

- **Vecino más cercano (Zaliapin & Ben-Zion, 2008/2013):** el que usa este
  proyecto. Define una **distancia entre pares de sismos** y deja que **los datos
  digan dónde está el corte**. Es el corazón del pipeline y la sección 2 lo
  explica entero.

### 1.5. La pregunta concreta del proyecto

> **Dado un sismo principal y sus parámetros (magnitud, ubicación, momento),
> ¿cuántas réplicas va a tener?**

El pipeline de la Entrega 1 **no modela** eso todavía — construye el **dataset**
para poder modelarlo en la Unidad 3. Cada fila del dataset de salida es **una
secuencia sísmica (un cluster)**: un sismo principal con las réplicas y
premonitores que el método le atribuyó. La columna objetivo es `n_aftershocks`.

---

## 2. El método Zaliapin & Ben-Zion (el corazón científico)

### 2.1. La métrica η (eta): distancia entre dos sismos

Para un sismo `j` y un candidato a padre `i` **anterior** a él:

```
η_ij  =  t_ij  ·  (r_ij)^d  ·  10^(−b · m_i)
         └──┬──┘   └───┬───┘   └─────┬──────┘
          tiempo    distancia    tamaño del PADRE
         (en años)  epicentral
                     (en km)
```

Tres factores, cada uno con su intuición:

| Factor | Lectura |
|---|---|
| `t_ij` | Cuánto tiempo pasó entre los dos. Más tiempo → menos parece réplica. |
| `(r_ij)^d` | Distancia entre epicentros, elevada a la dimensión fractal `d`. Corrige por lo amontonados que están los sismos *en general* en esa región: donde todo está apelotonado, 10 km es "lejos"; donde está disperso, no. |
| `10^(−b·m_i)` | El tamaño del **padre**. Un sismo grande genera réplicas más lejos y durante más tiempo, así que este factor **encoge** las distancias a su alrededor. La `m` que entra es la del **padre `i`, no la del hijo `j`**. |

Ese último punto hay que tenerlo clarísimo: un **M7 puede "reclamar" como hijo**
a un sismo que ocurrió tres semanas después y a 200 km, mientras que un **M4.5
solo alcanza** a lo que le pasa al lado y enseguida.

**Distancia epicentral, no hipocentral** (ignora la profundidad): la profundidad
tiene un error mucho mayor que el epicentro y ensuciaría el escaleo (`geo.py`).

### 2.2. Vecino más cercano → el "bosque de padres"

A cada sismo se le busca, **entre todos los anteriores**, el padre que
**minimiza η**.

- η mínimo chico → ese sismo tenía a alguien muy cerca en la métrica → **parece
  réplica**.
- η mínimo grande → apareció solo → **parece independiente**.

Resultado: cada evento tiene **exactamente un padre** (o ninguno, si es de los
primeros del catálogo). Eso arma un **bosque**: árboles donde la raíz es un
evento independiente y las réplicas cuelgan, posiblemente **en cadena** (réplica
de una réplica).

Este paso **todavía no decide nada**: solo tiende el árbol. Guarda `parent_id` y
`eta` para cada evento.

Costo: **O(N²)** — un catálogo de 20.000 eventos son 400 millones de pares. El
código lo hace **por bloques de filas** (256 por defecto) acumulando el mínimo,
en vez de materializar la matriz completa (que serían ~3 GB). Ver `neighbors.py`
y `geo.py`.

### 2.3. La bimodalidad: dónde cortar lo dicen los datos

Si hacés el **histograma de `log₁₀(η)`** sobre un catálogo real, aparecen **dos
jorobas**:

```
   frecuencia
      │      ╱╲                    ╱╲
      │     ╱  ╲                  ╱  ╲
      │    ╱    ╲       valle    ╱    ╲
      │   ╱ RÉPLICAS ╲    ↓     ╱ FONDO ╲
      │  ╱  (η bajo)   ╲______╱  (η alto) ╲
      └────────────────────────────────────── log₁₀(η)
```

Eso **no está impuesto por el método** — sale de los datos, y es la evidencia de
que hay **dos poblaciones** conviviendo en el catálogo. El umbral `η₀` se pone en
el valle.

**Cómo se ajusta el umbral (no a ojo):** se le pone una **mezcla de dos
gaussianas** a `log₁₀(η)` por el algoritmo **EM (Expectation-Maximization)**, y
`η₀` queda donde las dos componentes pesadas se cruzan (el punto de **mínimo
error de clasificación**). Ver `threshold.py`.

Detalle de implementación importante para el oral: **el EM está escrito a mano**
(30 líneas), no traído de scikit-learn, "para que quede a la vista qué se está
haciendo". Arranca con las medias en los cuartiles (p25 y p75) en vez de al azar
→ **resultado reproducible sin semilla**.

**Diagnósticos que el ajuste reporta** (y por qué):

- `separation_sigmas`: distancia entre las dos medias en unidades de desvío. Si
  es < 1.5, **avisa**: los modos se pisan y `η₀` es frágil (pasa con catálogos
  chicos o que mezclan regiones).
- Si una componente se queda con < 5% de los datos → **avisa**: "el EM no
  encontró dos poblaciones, encontró una y un rincón".
- `expected_classification_error`: cuánta réplica queda del lado del fondo y
  viceversa, según el propio modelo.

### 2.4. La factorización T·R

Zaliapin & Ben-Zion parten η en dos mitades independientes:

```
T_ij = t_ij · 10^(−q·b·m_i)        (tiempo reescalado)
R_ij = (r_ij)^d · 10^(−p·b·m_i)    (distancia reescalada)
```

con `p + q = 1` (acá `p = q = 0.5`), de forma que **η = T · R**. En el plano
`(log T, log R)` las dos poblaciones se ven separadas **a simple vista**, a 45°.
Por eso el código guarda `log10_T` y `log10_R` además de `log10_eta`.

### 2.5. Thinning probabilístico: de la línea dura al sorteo

El umbral `η₀` parte la población con **una línea**, pero esa línea **miente en
los bordes**: un evento que cae justo por debajo no es "más réplica" que uno que
cae justo por encima.

El **thinning** reemplaza la línea por un **sorteo**:

1. A cada evento se le calcula `p_background` = probabilidad de ser fondo = **la
   responsabilidad posterior** de la componente del fondo en la mezcla ajustada.
2. Se sortea: `is_aftershock = (random() ≥ p_background)`.
3. Los que están lejos del umbral casi no se mueven; los del medio se reparten.

De ahí sale la columna `is_aftershock`. Se guarda también `is_aftershock_eta` (el
corte duro) para poder comparar cuánto movió el thinning.

**Como el thinning es un sorteo, es estocástico** → la semilla (`seed`) es un
parámetro del DAG y queda escrita en el nombre de los archivos. Sin fijarla, los
números del informe no se reproducen.

**Desviación consciente del paper:** Zaliapin & Ben-Zion estiman el peso del
fondo comparando contra el catálogo barajado. Acá `p_background` sale de la
**mezcla gaussiana**, no del barajado, porque sobre este catálogo el barajado no
identifica bien ese peso (conserva N → tiene el doble de densidad que el fondo
que quiere modelar). En un control con 50% de réplicas plantadas, el barajado
devuelve 0.96 en vez de 0.50; el posterior de la mezcla recupera el 50% con
94.6% de precisión.

### 2.6. El catálogo nulo (randomización): ¿la bimodalidad es real?

Pregunta que ningún ajuste puede contestar solo: **¿las dos jorobas son del
catálogo, o las fabrica la métrica η?**

Se construye un **catálogo nulo**: se **barajan los tiempos y las magnitudes**
entre sí y **se dejan los epicentros donde están**. Eso:

- **destruye** la asociación temporal (no hay más réplicas),
- **conserva** la geografía (que responde a los bordes de placa, no a las
  réplicas).

Da la distribución de η **"si no hubiera réplicas en absoluto"**. Si el catálogo
barajado **también** se parte en dos modos y cae bajo el umbral la **misma
proporción** de eventos → lo que estás marcando como réplicas es solo la forma
que tiene η cuando no pasa nada.

Es **la parte cara**: cada barajada es una corrida completa del vecino más
cercano (O(N²)). Se hacen 5 por defecto. También es estocástico → semilla en el
nombre del archivo.

`contrast_with_null()` reporta el **exceso** (fracción bajo umbral en el real −
en el nulo) y **avisa si es ≤ 5 puntos**. (El README reconoce que esto **debería
ser un cociente, no puntos absolutos** — está en "Lo que falta".)

### 2.7. Reconstrucción de clusters: de enlaces sueltos a secuencias

Quedarse con los enlaces aceptados por el thinning deja el bosque. Pero
**`groupby("parent_id")` cuenta solo hijos directos**, y las réplicas tienen
réplicas:

```
A (independiente)
├── B  (réplica de A)
│   └── D  (réplica de B, pero pertenece al cluster de A)
└── C  (réplica de A)
```

`groupby("parent_id")` diría "A tiene 2". Lo cierto: **el cluster de A tiene 3
réplicas** y B no es principal de nada.

El recorrido del bosque sale **O(N), sin recursión, sin union-find**, gracias al
**contrato de silver** (catálogo ordenado por tiempo → todo padre es anterior a
su hijo → una sola pasada hacia adelante: al llegar a un evento, el cluster de su
padre ya está resuelto).

Produce **dos tablas** (en parquet; la tarea `publish_csv` las vuelve a escribir
en CSV al final — ver 2.8):

- **`events_*`**: catálogo evento por evento con `cluster_id`, `generation`,
  `order_in_cluster`, `is_mainshock`.
- **`summary_*`**: **una fila por cluster** — este es **el entregable
  principal**. Columnas: `mainshock_id/mag/time/lat/lon`, `n_events`,
  `n_aftershocks`, `n_foreshocks`, `duration_days`, `extent_km`,
  `max_generation`, `root_is_mainshock`.

**Quién es el sismo principal es una decisión de método** (parámetro `mainshock`):

- `"largest"`: el de mayor magnitud del cluster (lo que usan Zaliapin & Ben-Zion).
- `"root"`: el que disparó la secuencia.

Difieren cuando hubo premonitores (un M4.5 abre el árbol, 3 horas después llega
el M7). En la corrida de referencia discrepan en **42 clusters (2.1%)**.

**Control:** `productivity()` ajusta la ley de Utsu (`nº medio de réplicas ~
10^(α·M)`) sobre el resumen, **incluyendo los clusters de cero réplicas**. Si `α`
cae fuera de [0.5, 1.5], **avisa**: el problema está aguas arriba (Mc, umbral,
thinning).

### 2.8. Publicación y validación: `publish_csv` / `delivery.py`

Es el **último paso** del pipeline y la **única tarea que escribe CSV** (todas
las demás usan parquet, que se lee tipado entre tareas pero no se abre con doble
clic). Módulo: `include/sismos/delivery.py`.

**Publica las dos unidades de análisis posibles del proyecto**, en
`include/output/detector_replicas_api/delivery/`:

| Archivo | Una fila es… | Clave | Objetivo |
|---|---|---|---|
| `clusters_*.csv` | una **secuencia sísmica**: un sismo principal con todas sus réplicas | `cluster_id` | `n_aftershocks` |
| `events_*.csv` | un **terremoto**, con a qué secuencia cayó | `id` | `is_aftershock` |

**Y valida antes de dar la corrida por buena.** `check()` corre sobre cada
tabla los criterios medibles de la consigna, y si alguno se viola **la tarea
falla** (queda en rojo en la interfaz):

- clave con duplicados,
- menos de `MIN_ROWS = 1000` filas,
- menos de `MIN_COLUMNS = 5` columnas,
- alguna columna **entera** en nulo (eso no es "un nulo", es una columna que no
  se calculó).

Los **nulos normales no se penalizan** (un dataset real los tiene) pero **sí se
reportan**: el log deja la proporción de nulos por columna y el conteo de tipos
— justo los números que hay que tener a mano en la defensa.

Dos decisiones de diseño para poder defender:

1. **Escribe el CSV *antes* de validar.** Si el chequeo falla, querés poder abrir
   el archivo y ver *por qué* falló, no quedarte sin nada.
2. **Un dataset roto publicado en verde es peor que una corrida en rojo**, porque
   el error aparece recién cuando alguien ya construyó algo encima. Es la misma
   idea que una tarea de validación en cualquier pipeline de producción: el
   último paso antes de exponer el dato es dudar de él.

---

## 3. La arquitectura técnica

### 3.1. Orquestación: Airflow 3 sobre Astronomer

- **Astro Runtime 3.3-4** (imagen `astrocrpublic.azurecr.io/runtime:3.3-4`),
  **Airflow 3**, corriendo en **Docker**.
- `cd pipeline && astro dev start` → interfaz en `http://localhost:8080`.
- El DAG `detector_replicas_api` **no tiene schedule** (`schedule=None`): se
  dispara a mano con *Trigger DAG w/ config*, porque **la ventana de fechas la
  elige quien lo corre**.
- La conexión `usgs_fdsn` (host `earthquake.usgs.gov`, https) se crea sola desde
  `airflow_settings.yaml` — ese archivo **se versiona** porque no tiene secretos
  (el FDSN es abierto).
- Se usa la **TaskFlow API** (`@dag`, `@task`): las funciones decoradas se llaman
  entre sí y Airflow arma el grafo de dependencias a partir de eso.

### 3.2. Dónde vive la lógica y por qué

| Ruta | Qué hay |
|---|---|
| `dags/detector_replicas_api.py` | **Solo orquestación**: params, caché, dependencias entre tareas. |
| `include/sismos/` | **La lógica importable** (`import sismos`, vía `PYTHONPATH` del Dockerfile). |
| `include/output/` | Salida de cada capa. Gitignored: se regenera corriendo el DAG. |
| `tests/dags/` | Tests de integridad del DAG. |

La lógica va en `include/` y no en `dags/` por dos razones concretas: **se testea
sin levantar Airflow**, y **los notebooks del repo pueden importar exactamente el
mismo código que corre en producción**.

### 3.3. Arquitectura medallón (bronze → silver → gold)

Es el patrón de capas de data engineering. Mapea directo a las preguntas de
ENTREGA1 ("capa plata" / "capa bronce"):

| Capa | Formato | Qué es | Regla |
|---|---|---|---|
| **bronze** (`bronze/*.csv.gz`) | CSV crudo, gzipeado | El CSV **tal cual vino de la API**, sin castear una coma (`dtype=str`). | "Espejo de la fuente." Si mañana descubrís que interpretabas mal una columna, la arreglás sin volver a pedirle nada al USGS. |
| **silver** (`silver/*.parquet`) | Parquet tipado | El catálogo **limpio**: fechas parseadas, números casteados, filas sin epicentro/tiempo/magnitud descartadas, deduplicado por `id` (se queda con el `updated` más reciente), **ordenado por tiempo**. | **Solo higiene, nada de método.** Por eso se calcula una vez y se reutiliza aunque cambies el umbral o la semilla. El orden cronológico es **parte del contrato**. |
| **gold** (`neighbors/`, `threshold/`, `null/`, `aftershocks/`, `clusters/`) | Parquet / JSON | Lo que produce el método, paso por paso. | Lo que depende de la semilla la lleva en el nombre. |
| **delivery** (`delivery/*.csv`) | **CSV** | El entregable: `clusters_*.csv` (una fila por secuencia) y `events_*.csv` (una fila por terremoto). Lo escribe y valida `publish_csv`. | Es la única capa en CSV, porque es la que se abre a mano para mirarla y defenderla. |

### 3.4. Particionado estilo Hive + caché idempotente

`sismos/__init__.py` → `partition()` es **la única fuente de verdad de qué
identifica a una corrida**. Genera nombres tipo:

```
earthquake_starttime=2016-01-01_endtime=2026-01-01_minmagnitude=3.5_bbox=-55,-21,-74,-53.parquet
```

Todo lo que cambie los datos de entrada **tiene que aparecer en el nombre**. Si
no, dos consultas distintas se pisan el mismo archivo y "estás analizando
Argentina creyendo que es California". Las capas que dependen de la semilla le
agregan `_seed=N`.

**Caché:** cada tarea, antes de recalcular, chequea si su salida ya existe **y es
más nueva que su insumo** (`st_mtime`). El parámetro `force=True` ignora todas
las cachés. Esto hace el pipeline **idempotente**: correrlo dos veces con los
mismos params da el mismo resultado y no vuelve a pegarle a la API.

`land_bronze` está **separado del resto** a propósito: es lo único que puede
fallar por razones ajenas al código (API caída, lenta, cambiada). Si el CSV ya
está en disco, no se vuelve a pedir → podés iterar sobre todo el resto sin
castigar al USGS.

### 3.5. El tope de 20.000 del FDSN y la bisección del rango

El servicio **rechaza con HTTP 400** cualquier consulta que empareje **más de
20.000 eventos**. Pasarle el parámetro `limit` sería peor: como ordena por tiempo
descendente, te devolvería los 20.000 **más recientes** y el CSV cubriría una
ventana más corta que la pedida **sin decírtelo** — el `starttime` del nombre del
archivo estaría mintiendo, y Mc/b/d saldrían sobre un catálogo recortado a
escondidas.

Cómo lo resuelve `usgs_earthquake.py` (`fetch()`):

1. Primero llama al endpoint `count` para preguntar **cuántos eventos hay** sin
   bajarlos.
2. Si `limit` está seteado y el total lo supera → **falla con un error claro**
   ("subí `limit`, acortá la ventana o subí `minmagnitude`"). El `limit=20000`
   por defecto es un **tope de seguridad**, no un recorte.
3. Si el total entra en el tope del servicio (`SERVICE_LIMIT = 20000`) → una
   sola request.
4. Si no entra → **`_download()` bisecta el rango de fechas recursivamente**: parte
   la ventana por la mitad, cuenta cada mitad, y repite hasta que cada pedido
   entre. `_concatenate()` pega los CSV de los pedazos dejando **una sola
   cabecera** (si no, quedarían filas con la palabra `time` en el medio y silver
   las volvería nulos). Un evento en el borde entre dos sub-ventanas puede venir
   repetido → silver deduplica por `id`.

**Por qué bisecta en vez de repartir en partes iguales:** los sismos no se
distribuyen parejo en el tiempo — una secuencia de réplicas mete miles de
eventos en un par de días. Un reparto uniforme dejaría ese pedazo igual de
grande que el original; cortando por la mitad y volviendo a contar, la partición
se **adapta a dónde está la densidad**. Cada nivel de la recursión cuesta *una*
consulta de conteo (el total del padre ya viene contado).

Otros detalles: reintentos espaciados (`_request`); el **rectángulo**
(`minlatitude/maxlatitude/minlongitude/maxlongitude`, por defecto Argentina
continental) lo aplica **el propio servicio**, no un filtro post-descarga.

> Nota: para la corrida de referencia (Argentina, 10 años, M≥3.5 → 7.320
> eventos) la bisección nunca se dispara porque está muy por debajo de 20.000.
> Sirve para ventanas más largas o regiones más grandes. *(Implementado en el
> commit `2a72f9d`; antes el README lo describía pero el código hacía una sola
> request.)*

### 3.6. Por qué el rectángulo (bbox) es la decisión que más pesa

El método **supone un catálogo homogéneo**. El mundo entero mezcla regiones con
completitudes muy distintas, le da a `d` una geometría que responde a los bordes
de placa (no a una ley de potencias), y hace que los árboles de réplicas
**encadenen zonas sin relación** hasta armar clusters de 5.000 km. Por eso el
default es Argentina continental (`bbox=-55,-21,-74,-53`).

### 3.7. Detalles numéricos

- **Haversine** para distancias sobre la esfera (`geo.py`), vectorizado por
  bloques.
- `R_MIN_KM = 0.01`: piso para la distancia, así dos eventos con el mismo
  epicentro no dan `r=0 → η=0` (ese padre ganaría siempre).
- **Mc por máxima curvatura (MAXC):** el bin más poblado de la distribución
  frecuencia-magnitud **no acumulada**, + una corrección empírica de **+0.2**
  (Woessner & Wiemer 2005, porque MAXC tiende a subestimar). El binning es de 0.1
  (la grilla en que el USGS publica magnitudes).
- **`b` y `d` NO se estiman**, se toman estándar (1.0 y 1.5). Para `d` es incluso
  más defendible: el ajuste por integral de correlación necesita elegir un rango
  de escaleo y resultó frágil (daba 0.9 en California, donde el valor publicado
  es ~1.6). Como `b` y `d` entran en η como una **reescala** y el umbral se
  ajusta *después* sobre la distribución que salga, esto **no invalida el
  método** — solo perdés poder decir que los parámetros son "de estos datos".
- Los 3 parámetros (`mc`, `b`, `d`) **no se persisten**: son tan baratos (un
  histograma y un argmax) que viajan por **XCom** hasta los pasos que los
  necesitan.

---

## 4. Tarea por tarea del DAG

El grafo (`tests/dags/test_dag_integrity.py` lo fija explícitamente):

```
land_bronze
   └─> refine_silver
          ├─> estimate_mc ──────────────┐
          ├─> nearest_neighbor <────────┤ (usa silver + mc/b/d)
          │      ├─> fit_eta_threshold  │
          │      └─> thinning <─────────┼──── (cruza vecinos + umbral + nulo)
          └─> randomize_catalog <───────┘
                 └─> thinning
                        └─> build_clusters
                               └─> publish_csv   ← hoja / entregable
```

| Tarea | Insumo | Qué hace | Técnica |
|---|---|---|---|
| **`land_bronze`** | params de consulta | Baja el CSV crudo de la API, lo gzipea, lo guarda particionado. No lo toca. | `count` + `query` del FDSN, bisección del rango, reintentos |
| **`refine_silver`** | bronze | Parsea fechas/números, tira filas sin epicentro/tiempo/magnitud, deduplica por `id`, **ordena por tiempo**. | pandas: `to_datetime(ISO8601)`, `to_numeric`, `drop_duplicates`, `sort_values` |
| **`estimate_mc`** | silver | Estima **Mc** por máxima curvatura; devuelve `{mc, b, d}` (b y d estándar). Corta la corrida si quedan < 50 eventos sobre Mc. | FMD + argmax + corrección 0.2 |
| **`nearest_neighbor`** | silver + `{mc,b,d}` | Recorta a Mc, calcula η para todos los pares, asigna a cada evento su padre. Guarda `parent_id`, `eta`, `T`, `R`. | O(N²) por bloques, haversine vectorizada |
| **`fit_eta_threshold`** | vecinos | Ajusta mezcla de 2 gaussianas a `log₁₀(η)`, pone `η₀` en el cruce. Reporta separación de modos y error esperado. | **EM a mano**, cruce por grilla + interpolación |
| **`randomize_catalog`** | silver + `{mc,b,d}` | Baraja tiempos y magnitudes (epicentros fijos) N veces, corre el vecino más cercano en cada una → distribución nula de η. | permutaciones + N× O(N²). Semilla en el nombre |
| **`thinning`** | vecinos + umbral + nulo | Sortea qué enlaces vuelven al fondo según `p_background` (posterior de la mezcla). Produce `is_aftershock`. Contrasta contra el nulo. | Bernoulli por evento, semilla `seed+1` |
| **`build_clusters`** | réplicas clasificadas | Recorre el bosque en O(N), arma clusters, cuenta réplicas/premonitores por sismo principal. Produce `events_*` y `summary_*` (parquet). Control: ley de Utsu. | Pasada cronológica única, haversine a un punto, `polyfit` para α |
| **`publish_csv`** | eventos + resumen | Escribe los dos **CSV** finales (`clusters_*.csv`, `events_*.csv`) en `delivery/` y los **valida**: clave única, ≥1000 filas, ≥5 columnas, sin columnas vacías. Si algo falla, la tarea queda en rojo. | pandas `to_csv`, `check()` (`delivery.py`) |

**Parámetros del DAG** (los ves en *Trigger w/ config*):

- Consulta: `starttime`, `endtime`, `minmagnitude`,
  `minlatitude/maxlatitude/minlongitude/maxlongitude` (bbox), `limit`.
- Método: `mainshock` (`largest`/`root`), `seed`, `n_randomizations`.
- `force`: ignora todas las cachés.

---

## 5. El dataset de salida y los 7 criterios de ENTREGA1

`publish_csv` escribe **dos** CSV en `include/output/detector_replicas_api/delivery/`:

| Archivo | Una fila es… | Clave | Objetivo | Filas (corrida ref.) |
|---|---|---|---|---|
| **`clusters_*.csv`** *(entregable principal)* | una **secuencia sísmica**: un sismo principal con todas sus réplicas y premonitores atribuidos | `cluster_id` | `n_aftershocks` | **1.978** |
| `events_*.csv` | un **terremoto**, con a qué secuencia cayó | `id` | `is_aftershock` | 2.321 |

**"Una fila es un ___"** (para el `clusters_*.csv`) → *"una secuencia sísmica: un
sismo principal junto con las réplicas y premonitores que el método del vecino
más cercano le atribuyó"*. Coincide con la pregunta del proyecto (predecir
réplicas de un sismo). **Columna objetivo** → `n_aftershocks`.

Con los números de la **corrida de referencia** (Argentina, `starttime`
2016-01-01, `endtime` 2026-01-01, `minmagnitude` 3.5, `seed` 1,
`n_randomizations` 5):

| Criterio | Qué tiene que dar | En `clusters_*.csv` |
|---|---|---|
| **Clave sin duplicados** | `df["cluster_id"].is_unique` → `True` | ✅ **1.978 claves únicas**. `cluster_id` = id del evento raíz del cluster; los ids del USGS son únicos y silver deduplica. **`publish_csv` falla si repite.** |
| **Volumen** | > 1.000 filas | ✅ **1.978 clusters** (131 con ≥1 réplica). **`publish_csv` falla si hay < 1.000.** |
| **Ancho** | ≥ 5 columnas útiles | ✅ **15 columnas**. `publish_csv` falla si hay < 5. |
| **Mezcla de tipos** | numéricas + categóricas + fechas | ✅ numéricas (`mainshock_mag`, `lat`, `lon`, `n_aftershocks`, `duration_days`, `extent_km`, `max_generation`), fecha (`mainshock_time`), texto/categórica (`cluster_id`, `mainshock_id`, `mainshock_place`), booleana (`root_is_mainshock`). `publish_csv` loguea `df.dtypes.value_counts()`. |
| **Nulos conocidos** | saber cuáles y por qué | El resumen sale **sin nulos** en la corrida de referencia. Los clusters unitarios tienen `duration_days=0` y `extent_km=0` — **ceros legítimos, no nulos.** `publish_csv` loguea la proporción de nulos por columna. |
| **Sin columnas 100% nulas** | ninguna | ✅ **`publish_csv` falla si alguna columna quedó entera en nulo.** |
| **Bronze guardado** | ¿queda el dato crudo? | ✅ `bronze/*.csv.gz` es la respuesta de la API tal cual llegó |

**Lo importante del cambio de `2a72f9d`:** estos seis criterios ya **no dependen
de que vos los verifiques a mano** — los chequea el propio pipeline en
`publish_csv` y, si alguno falla, **la corrida queda en rojo**. Eso es
exactamente lo que pide la consigna ("ahora la calidad del dato es tuya").

**Reproducibilidad ("¿sale el mismo archivo si lo corrés de nuevo?")**: **sí**,
dados los mismos params + la misma `seed`. Las únicas partes estocásticas son
`randomize_catalog` y `thinning`, y las dos están sembradas. El EM del umbral es
determinístico (inicia en los cuartiles, sin azar).

**La "concesión" a declarar (equivalente al `player_positions` del dataset
canónico):**

1. **`b` y `d` son constantes**, no ajustados a estos datos (el README lo suma
   explícitamente a "Lo que falta").
2. **El thinning usa la mezcla gaussiana, no el barajado**, para el peso del
   fondo (desviación consciente del paper).
3. **Todavía hay clusters imposibles**: 2 se extienden > 1.500 km (el mayor,
   **3.610 km**) porque el rectángulo de Argentina abarca ~3.800 km y un M grande
   en Jujuy puede reclamar un hijo en Tierra del Fuego. Falta cortar por
   profundidad. El pipeline **no los esconde**: los cuenta y avisa por log.
4. **El contraste con el nulo usa puntos absolutos en vez de un cociente**
   (documentado en "Lo que falta").

---

## 6. Cómo se valida que las cuentas están bien

### La corrida de referencia (memorizar estos números)

Argentina continental, `starttime` 2016-01-01, `endtime` 2026-01-01,
`minmagnitude` 3.5, `seed` 1, `n_randomizations` 5. *"Con estos parámetros salen
exactamente estos números."*

| | |
|---|---|
| Eventos en el catálogo limpio (silver) | 7.320 |
| Mc (máxima curvatura) | 4.50 |
| Eventos completos, sobre Mc | 2.321 |
| `b` y `d` | 1.0 y 1.5, estándar |
| Umbral | log₁₀ η₀ = −5.778 (modos separados 2.01 σ, error esperado 4.6%) |
| Contraste con el catálogo barajado | 12.2% bajo el umbral contra 6.2% — exceso de 6.0 puntos |
| Réplicas después del thinning | 343 de 2.321 (14.8%) |
| Clusters | 1.978, de los cuales 131 tienen ≥1 réplica |
| Cluster más grande | 32 réplicas |
| Raíz ≠ sismo principal | 42 clusters (2.1%) — secuencias con premonitores |
| Productividad de Utsu | α = 1.227 (R² 0.991) sobre 5 bins |
| Entregable (`clusters_*.csv`) | 1.978 filas × 14 columnas, sin nulos, sin claves repetidas |

### Cada paso contra un caso de respuesta conocida

El `README.md` tiene la tabla completa. Los conceptualmente importantes:

- **Vecino más cercano**: contra un bucle O(N²) ingenuo sobre 300 eventos → **0
  padres distintos**. Y con bloques de 7 vs 256 → **idéntico** (el chunking no
  cambia el resultado).
- **Bimodalidad**: sobre un sintético **sin** réplicas → unimodal (no inventa un
  segundo modo). Con **50%** de réplicas plantadas → detecta 48.4%.
- **Thinning**: sobre un control con 50% de réplicas → marca 49.6%, **precisión
  94.6%, recall 93.8%**.
- **Clusters**: bosque armado a mano (cadena A→B→D + un premonitor) → topología
  exacta. E **invariantes** sobre el catálogo real: los eventos por cluster suman
  el total, cada evento cae en exactamente un cluster, `réplicas + premonitores +
  1 = tamaño`.
- **Productividad**: el único control **contra un hecho externo** (no un
  sintético). La ley de Utsu se reproduce con `α = 1.227`, `R² = 0.991` sobre 5
  bins de magnitud — dentro del rango que se observa en el mundo.
- **Máxima curvatura acertó Mc = 4.50**: coincide con lo que se ve en los
  conteos crudos del USGS para Argentina (la cantidad de eventos es casi igual
  pidiendo M≥2.5, 3.0, 3.5, 4.0, y recién cae en 4.5 → ese tramo plano es el
  catálogo global al que le faltan los sismos chicos que sí tiene el INPRES).

---

## 7. Puntos abiertos y su estado

Este documento marcaba cuatro cosas que faltaban. El commit `2a72f9d` cerró tres
de ellas. Estado actual:

| # | Punto | Estado |
|---|---|---|
| 7.1 | El DAG no parseaba (`NameError`) | ✅ **Resuelto** en `2a72f9d` |
| 7.2 | La salida era parquet, no CSV | ✅ **Resuelto** en `2a72f9d` (`publish_csv`) |
| 7.3 | La bisección del rango estaba documentada pero no codeada | ✅ **Resuelto** en `2a72f9d` |
| 7.4 | El docstring del DAG estaba desactualizado | ✅ **Resuelto** en `2a72f9d` |
| 7.5 | Ventana de fechas: tiene que ser larga | ⚠️ No es un bug, pero hay que saberlo |
| 7.6 | Target inflado de ceros (Unidad 3) | 🔜 Sigue abierto, es para más adelante |
| 7.7 | Números inconsistentes dentro del `README.md` | ⚠️ Menor, conviene alinearlos antes del oral |

### 7.1. El DAG ya parsea *(resuelto)*

Había un bloque muerto que llamaba a `estimate_mc_b_d` (nombre viejo de la tarea
`estimate_mc`), quedado de un mal rebase. Tiraba `NameError` al registrar el DAG
→ "Broken DAG" en la interfaz y `test_no_import_errors` en rojo. `2a72f9d` borró
ese bloque; el cableado bueno (el que usa `estimate_mc`) ya estaba abajo.
`astro dev pytest` ahora pasa.

### 7.2. La salida ya es CSV *(resuelto)*

La tarea `publish_csv` + el módulo `sismos/delivery.py` escriben
`delivery/clusters_*.csv` y `delivery/events_*.csv`, y los validan (ver 2.8 y la
sección 5). Antes todas las capas eran parquet y no había CSV final.

### 7.3. La bisección del rango ya está implementada *(resuelto)*

`_download()` en `usgs_earthquake.py` parte el rango de fechas por la mitad
recursivamente cuando la consulta emparejaría más de 20.000 eventos, y
`_concatenate()` pega los CSV. Ver 3.5. Para la corrida de referencia no se
dispara (7.320 eventos), pero ya no es "documentado pero no hecho".

### 7.4. El docstring del DAG ya está actualizado *(resuelto)*

Ya no dice "EN CONSTRUCCIÓN". Describe el método completo y aclara que **una fila
del entregable es una secuencia sísmica**, clave `cluster_id`, objetivo
`n_aftershocks`.

### 7.5. La ventana de fechas tiene que ser larga

`estimate_mc` **corta la corrida** si quedan menos de **50 eventos por encima de
Mc** (`MIN_EVENTS`). El `pipeline/README.md` lo dice explícito: *"con cinco
semanas de Argentina quedan ocho y falla; con diez años, sobran"*. Usá siempre la
ventana de referencia:

```
starttime = 2016-01-01   endtime = 2026-01-01   minmagnitude = 3.5   seed = 1
```

Si además `publish_csv` ve menos de 1.000 filas, **también falla** — así que una
ventana corta te va a dar la corrida en rojo por dos lados.

### 7.6. Target inflado de ceros (Unidad 3)

El target `n_aftershocks` está **fuertemente inflado de ceros**: 1.978 clusters,
solo 131 con réplicas (≈93% de ceros). Para modelar vas a necesitar algo que lo
maneje (Poisson/binomial negativa con inflado de ceros, o un modelo en dos
etapas). No afecta la Entrega 1.

### 7.7. Inconsistencias menores en el `README.md`

`2a72f9d` actualizó la tabla de la corrida de referencia pero dejó dos números
viejos en el texto en prosa:

- Sección `### build_clusters`: *"discrepan en el 4.6% de los clusters"* — la
  tabla de referencia dice **2.1% (42 clusters)**.
- Sección "Una corrida de referencia" (prosa): *"Que R² dé 0.996"* — la tabla
  dice **R² 0.991**.

No es grave, pero si en el oral alguien cita un número y otro cita el otro, queda
mal. Conviene alinear la prosa con la tabla (los valores buenos son los de la
tabla: 2.1% y 0.991).

---

## 8. Cómo mapear esto a las preguntas del oral

| Pregunta de ENTREGA1 | Respuesta corta |
|---|---|
| La pregunta del proyecto en una frase | "Dado un sismo principal y sus parámetros, ¿cuántas réplicas tendrá? Para eso necesitamos primero separar réplicas de sismos de fondo (declustering)." |
| La fuente | Servicio FDSN de eventos del USGS, endpoint `query`, acotado a un rectángulo (Argentina continental). |
| Qué es una fila | En `clusters_*.csv` (el entregable): una secuencia sísmica / cluster — un sismo principal + sus réplicas y premonitores atribuidos. En `events_*.csv`: un terremoto. |
| Cuántas filas / cuál es la clave | **1.978 filas**; clave `cluster_id` (id del evento raíz), única — y `publish_csv` **falla** si repite. |
| Columna objetivo | `n_aftershocks` (en el `events_*.csv`, `is_aftershock`). |
| Dónde hay nulos y por qué | El `clusters_*.csv` sale **sin nulos** en la corrida de referencia; clusters unitarios tienen duración/extensión = 0 (ceros, no nulos). En `silver` se descartan las filas sin epicentro/tiempo/magnitud. `publish_csv` loguea la proporción de nulos por columna. |
| ¿Sale el mismo archivo si lo corrés de nuevo? | Sí, con los mismos params y la misma `seed`. Randomización y thinning son las únicas partes estocásticas y están sembradas. |
| ¿Dónde está el dato crudo (bronce)? | `include/output/.../bronze/earthquake_...csv.gz`, la respuesta de la API sin tocar. |
| ¿Quién verifica los 7 criterios? | El propio pipeline: la tarea `publish_csv` los chequea y deja la corrida en rojo si alguno falla. |
| La concesión de tu dataset | `b` y `d` fijos (no "de estos datos"); thinning basado en la mezcla y no en el barajado; 2 clusters de >1.500 km que el método encadena mal (el pipeline los cuenta y avisa); contraste con el nulo en puntos absolutos. |
| Por qué cada tarea está donde está | Sección 4 de este documento + el `README.md` de la raíz. |

---

## 9. Ampliación del dataset: más columnas, más filas y otra pregunta

*(Sección agregada; no cambia nada de lo anterior.)*

El resumen de la Entrega 2 tenía sólo dos features sin fuga (`mainshock_mag`,
`mainshock_depth`) y apenas 131 de 1.978 secuencias con réplicas (93,4 % de ceros). Para
llegar a la Entrega 3 se corrigió en cinco pasos, cada uno con sus tests y sin mover la
corrida de referencia (1.978 / 131 / 279):

1. **Tope de distancia padre-hijo** (`r_max_km`, `r_max_mode=gk`): evita el
   encadenamiento de sismicidad no relacionada (clusters de 3.600 km). Apagado por
   defecto.
2. **Sismicidad previa** (`n_prior_2d/30d/365d`, `prior_max_mag_365d`, `days_since_prior`)
   y el objetivo `has_aftershock`: columnas que existirían al predecir.
3. **Cinco regiones independientes** (Andes, Japón, Filipinas, Sumatra–Java, México–Centroamérica),
   cada una con su Mc y su η₀, concatenadas con una columna `region`: 19.570 secuencias,
   2.069 con réplicas, 401 con M>6. Se agrega el efecto borde (`dist_to_border_km`,
   `near_border`).
4. **Objetivos por horizonte** en `horizons_*.csv`: "parado a *t* horas del principal,
   ¿cuántas réplicas habrá hasta *T* días?", con censura marcada.
5. **Documentación** de qué es causal y qué no (`FEATURES.md`) y las correlaciones.

La pregunta del proyecto puede reformularse, entonces, de "cuántas réplicas tendrá esta
secuencia" a "dado lo que se sabe hasta *t*, ¿habrá más réplicas, y cuántas?". Detalle de
columnas, advertencias de causalidad y números por región en `FEATURES.md`.

## Referencias

- **Zaliapin, Gabrielov, Keilis-Borok & Wong (2008)** — *Clustering analysis of
  seismicity and aftershock identification*. La métrica η.
- **Zaliapin & Ben-Zion (2013)** — *Earthquake clusters in southern California
  I*. El thinning y la reconstrucción de clusters.
- **Aki (1965)** — estimación de `b` por máxima verosimilitud.
- **Wiemer & Wyss (2000)** — Mc por bondad de ajuste.
- **Woessner & Wiemer (2005)** — corrección de MAXC.
- **Grassberger & Procaccia (1983)** — dimensión de correlación.
- **Gardner & Knopoff (1974)** — declustering por ventanas fijas.
- **Utsu (1970)** — ley de productividad de réplicas.
