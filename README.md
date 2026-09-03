# Detector de réplicas sísmicas

Proyecto integrador de Ciencia de Datos. Toma los terremotos del catálogo del
USGS y separa los que ocurrieron por su cuenta de los que son réplicas de otro,
con el método del vecino más cercano de Zaliapin & Ben-Zion.

Para levantar el pipeline: [`pipeline/README.md`](pipeline/README.md). Este
archivo explica **qué hace cada paso y por qué está ahí**.

## El problema

Un catálogo sísmico es una lista plana: fecha, lugar, magnitud. Pero los
terremotos no son independientes entre sí. Después de uno grande vienen
cientos de réplicas, apretadas en el tiempo y alrededor del mismo lugar.

Eso arruina casi cualquier cuenta que uno quiera hacer. Si preguntás "¿cuántos
sismos por año hay en esta zona?", la respuesta está inflada por secuencias de
réplicas que en realidad son *un solo* evento con muchas colas. Separar unos de
otros se llama **declustering**, y es el paso previo a casi toda la estadística
sísmica.

## La idea del método

Los métodos clásicos usan una ventana fija: "todo lo que ocurra dentro de 50 km
y 10 días de un sismo de magnitud 6 es réplica". El problema es que esas
ventanas son tablas que alguien decidió una vez.

Zaliapin & Ben-Zion lo dan vuelta: definen una **distancia entre pares de
sismos** y dejan que los datos digan dónde está el corte. Para un sismo `j` y un
candidato a padre `i` anterior a él:

```
eta = t · r^d · 10^(-b · m_padre)
```

Tres factores, cada uno con su intuición:

| Factor | Qué dice |
|---|---|
| `t` | Tiempo entre los dos. Cuanto más pasó, menos parece réplica. |
| `r^d` | Distancia entre epicentros, corregida por lo apelotonados que están los sismos en general. En una región donde todo está amontonado, 10 km es "lejos"; en una dispersa, no. Eso es lo que calibra `d`. |
| `10^(-b·m_padre)` | El tamaño del padre. Un sismo grande genera réplicas más lejos y durante más tiempo, así que este factor **encoge** las distancias a su alrededor. |

Ojo con el último: la magnitud que entra es la del **padre**, no la del hijo.
Es lo que hace que un M7 pueda "reclamar" como hijo a un sismo que ocurrió tres
semanas después y a 200 km, mientras que un M4.5 sólo alcanza a lo que le pasa
al lado y enseguida.

A cada sismo se le busca, entre **todos** los anteriores, el padre que minimiza
eta. Si el mínimo es chico, ese sismo tenía a alguien muy cerca → parece
réplica. Si es grande, apareció solo → parece independiente.

**Lo interesante:** si uno hace el histograma de `eta` sobre un catálogo real,
aparecen *dos jorobas*. Eso no está impuesto por el método — sale de los datos,
y es la evidencia de que hay dos poblaciones distintas conviviendo en el
catálogo. El umbral se ajusta ahí, en el valle entre las dos.

## Por qué cada tarea está donde está

El DAG es [`pipeline/dags/detector_replicas_api.py`](pipeline/dags/detector_replicas_api.py)
y la lógica vive en [`pipeline/include/sismos/`](pipeline/include/sismos/).

### `land_bronze` → CSV crudo, tal cual vino

Baja el catálogo de la API y lo guarda comprimido sin tocarle una coma. Está
separado del resto porque **es lo único que puede fallar por razones ajenas al
código**: la API se cae, cambia, tarda. Si el CSV ya está en disco no se vuelve
a pedir, así que se puede iterar sobre el resto del pipeline sin castigar al
USGS.

**El tope de 20000 del servicio.** El FDSN rechaza con HTTP 400 cualquier
consulta que empareje más de 20000 eventos. Pasarle su parámetro `limit` evita
el error, pero es peor: como ordena por tiempo descendente, devolvería los 20000
más recientes y el CSV cubriría una ventana más corta que la pedida sin decirlo
— el `starttime` del nombre del archivo estaría mintiendo y Mc, `b` y `d`
saldrían sobre un catálogo recortado a escondidas. Así que `fetch` pregunta
primero cuántos eventos hay y, si no entran, **parte el rango de fechas por la
mitad** hasta que cada pedido entre. Bisecta en vez de repartir en partes
iguales porque los sismos no se distribuyen parejo en el tiempo: una secuencia
de réplicas mete miles de eventos en un par de días.

El `limit` del DAG es otra cosa: un tope de seguridad. Si el rango empareja más
que eso, la corrida falla en vez de bajar un catálogo truncado. `0` es sin tope.

**El recorte a una región.** La consulta lleva un rectángulo
(`minlatitude`/`maxlatitude`/`minlongitude`/`maxlongitude`), que por defecto es
Argentina continental y lo aplica el propio servicio — filtrar después de bajar
sería pedir el mundo entero para tirar el 99%, y encima chocaría con el tope de
20000. Dejar los cuatro en null vuelve a consultar el planeta.

Es la decisión que más cambia los resultados, y no por gusto: el método supone
un catálogo homogéneo. El mundo entero mezcla regiones con completitudes muy
distintas, le da a `d` una geometría que responde a los bordes de placa y no a
una ley de potencias, y hace que los árboles de réplicas encadenen zonas sin
relación hasta armar clusters de 5000 km.

El rectángulo entra en el nombre de archivo de **todas** las capas, vía
`particion()`. Sin eso dos regiones distintas se pisarían el mismo `bronze` y
`silver`, y estarías analizando Argentina creyendo que es California.

### `refine_silver` → el catálogo limpio

Parsea fechas y números, tira las filas sin magnitud o sin epicentro, saca
duplicados y **ordena por tiempo**. Nada de método todavía, sólo higiene.

Está separado a propósito: silver no depende de ningún parámetro del algoritmo,
así que se calcula una vez y se reutiliza aunque después cambies el método de
Mc o el umbral. El orden cronológico es parte del contrato — el vecino más
cercano recorre el catálogo hacia atrás y lo da por sentado.

### `estimate_mc` → los tres números que el método necesita

De los tres, **uno solo se estima**.

**Mc, la magnitud de completitud.** Un catálogo registra todos los sismos
grandes y se le escapan los chicos: no hay sismógrafos en todos lados. Mc es la
magnitud a partir de la cual ya no se le escapa nada. Se calcula por máxima
curvatura: el bin más poblado de la distribución de magnitudes es donde el
catálogo deja de crecer y empieza a perder eventos. Éste no se puede tomar de
tabla, porque depende de qué red cubrió la zona y en qué época, y errarle sesga
todo lo que venga después.

**`b` y `d` se toman en sus valores estándar** (1.0 y 1.5). Son los que usa la
literatura cuando no se los ajusta. Para `d` es incluso *más* defendible que
ajustarlo: el ajuste por integral de correlación obliga a elegir un rango de
escaleo, y esa elección resultó frágil — sobre California devolvía 0.9 cuando el
valor publicado para esa región ronda 1.6.

Que sean constantes no invalida el método: entran en eta como una reescala, y el
umbral que separa réplicas de fondo se ajusta después sobre la distribución de
eta que salga. Lo que sí se pierde es poder decir que los parámetros son "de
estos datos".

Es tan barato —un histograma y un argmax— que no se persiste: los tres números
viajan por XCom hasta los pasos que los necesitan.

### `nearest_neighbor` → el bosque de padres

Calcula eta para todos los pares y le asigna a cada sismo su padre. Es O(N²) —
un catálogo de 20000 eventos son 400 millones de pares — así que va por bloques
acumulando resultados en vez de armar la matriz completa, que no entraría en
memoria.

Guarda `parent_id` y eta, pero también **T y R por separado**: eta se factoriza
en una mitad de tiempo y una de distancia (`eta = T · R`), y es en el plano
(T, R) donde las dos poblaciones se ven separadas a simple vista. Eso lo
necesita el paso siguiente.

Este paso **todavía no decide nada**: sólo tiende el árbol.

### `fit_eta_threshold` → dónde cortar

Le ajusta una mezcla de dos gaussianas al histograma de `log10(eta)` y pone el
umbral donde las dos se cruzan. El ajuste es EM escrito a mano en vez de traer
scikit-learn: son treinta líneas para dos gaussianas en una dimensión, y así
queda a la vista qué se está haciendo.

Reporta además **la separación entre los dos modos** y el error de clasificación
esperado, y avisa cuando el ajuste no se sostiene. Es importante: el método
presupone que la bimodalidad existe, y si el catálogo es chico o mezcla
regiones muy distintas, los dos modos se pisan y el umbral pasa a ser un número
frágil que conviene fijar a mano.

### `randomize_catalog` → un mundo donde no pasa nada

Baraja los tiempos y las magnitudes entre sí y deja los epicentros donde están.
Eso destruye la asociación temporal pero conserva la geografía del catálogo
—que responde a los bordes de placa, no a las réplicas— y da la distribución de
eta **si no hubiera réplicas en absoluto**.

Sirve para contestar la pregunta que ningún ajuste puede contestar solo: ¿la
bimodalidad es del catálogo, o la fabrica la métrica? Si el catálogo barajado
también se parte en dos modos y cae bajo el umbral la misma proporción de
eventos, entonces lo que se está marcando como réplicas es la forma que tiene
eta cuando no pasa nada.

Es la parte cara: cada barajada es una corrida completa del vecino más cercano.
Y es estocástica, así que la semilla es un parámetro del DAG y queda escrita en
el nombre de los archivos que produce.

### `thinning` → de la línea dura al sorteo

El umbral parte la población con una línea, pero esa línea miente en los
bordes: un evento que cae justo por debajo no es más réplica que uno que cae
justo por encima. El thinning reemplaza la línea por un sorteo: a cada evento
se le calcula la probabilidad de ser fondo y se lo devuelve al fondo con esa
probabilidad. Los que están lejos del umbral casi no se mueven; los del medio
se reparten. De ahí sale `is_aftershock`.

**Una desviación del paper que conviene tener presente.** Zaliapin & Ben-Zion
estiman el peso del fondo comparando la distribución observada contra la del
catálogo barajado. Acá la probabilidad sale de la mezcla de dos gaussianas ya
ajustada, no del barajado, porque sobre este catálogo el barajado **no
identifica ese peso**: al conservar N, tiene el doble de densidad que el fondo
que quiere modelar, sus vecinos caen más cerca y su eta se corre hacia abajo.
Sobre un control con 50% de réplicas plantadas el cociente entre distribuciones
devuelve 0.96 en vez de 0.50, y submuestrear el nulo para igualar densidades no
lo arregla: la ecuación de punto fijo o es degenerada o converge igual al valor
equivocado. El posterior de la mezcla, en cambio, recupera el 50% con 94.6% de
precisión.

### `build_clusters` → de enlaces sueltos a secuencias

Quedarse con los enlaces aceptados deja un bosque: cada evento independiente es
raíz de un árbol y las réplicas cuelgan de él, **en cadena**. Esa cadena es la
razón de ser del paso, porque agrupar por `parent_id` cuenta sólo los hijos
directos:

```
A (independiente)
├── B  (réplica de A)
│   └── D  (réplica de B, pero también del cluster de A)
└── C  (réplica de A)
```

`groupby("parent_id")` diría "A tiene 2, B tiene 1". Lo cierto es que el cluster
de A tiene tres réplicas y B no es sismo principal de nada. Con el decaimiento
de Omori las cadenas largas son la norma.

Recorrer el bosque sale gratis gracias al contrato de silver: como el catálogo
está ordenado por tiempo y todo padre es anterior a su hijo, una sola pasada
hacia adelante alcanza — al llegar a un evento, el cluster de su padre ya está
resuelto. O(N), sin recursión.

Produce dos tablas: el catálogo evento por evento con `cluster_id`,
`generacion`, `orden_en_cluster` e `is_mainshock`, y **el resumen por cluster**,
que es el entregable: sismo principal, magnitud, cantidad de réplicas,
premonitores, duración y extensión.

**Quién es el sismo principal** es una decisión, no un detalle. `mayor` toma el
de mayor magnitud (lo que usan Zaliapin & Ben-Zion) y `raiz` el que disparó la
secuencia. Difieren cuando hubo premonitores: un M4.5 abre el árbol y tres horas
después llega el M7. En la corrida de referencia discrepan en el 4.6% de los
clusters.

## Una corrida de referencia

Argentina continental, diez años (2016–2026), magnitud mínima de consulta 3.5:

| | |
|---|---|
| Eventos en el catálogo limpio | 7320 |
| Mc (máxima curvatura) | 4.50 |
| Eventos completos, sobre Mc | 2321 |
| `b` y `d` | 1.0 y 1.5, estándar |
| Umbral | log₁₀ η₀ = −5.778 (modos separados 2.01 σ) |
| Contraste con el catálogo barajado | 12.2% bajo el umbral contra 6.4% — exceso de 5.9 puntos |
| Réplicas después del thinning | 341 de 2321 (14.7%), semilla 1 |
| Clusters | 1980, de los cuales 126 tienen al menos una réplica |
| Cluster más grande | 33 eventos |
| Productividad de Utsu | α = 1.252 (R² 0.996) |

Tres cosas que vale la pena mirar:

**Máxima curvatura acertó el Mc.** Dio 4.50, y es exactamente lo que anticipaba
mirar los conteos crudos del USGS para Argentina: la cantidad de eventos es
prácticamente igual pidiendo M≥2.5, 3.0, 3.5 o 4.0, y recién cae en 4.5. Ese
tramo plano no es sismología, es un catálogo al que le faltan los sismos chicos
porque la red global no los ve — los tiene INPRES, no el USGS. Los 5000 eventos
que quedan por debajo de Mc se descartan, y está bien que así sea.

**La productividad reproduce la ley de Utsu.** Es el único control contra un
hecho externo, y no contra un sintético fabricado acá: el número medio de
réplicas por bin de magnitud sube de forma limpia y el exponente cae dentro del
rango que se observa en el mundo. Que R² dé 0.996 dice que los conteos por
magnitud son coherentes entre sí.

**Todavía hay clusters imposibles.** Tres se extienden más de 1500 km, el mayor
2808 km. El rectángulo de Argentina abarca 34° de latitud, casi 3800 km, así que
un sismo grande en Jujuy puede reclamar como hijo a uno en Tierra del Fuego. Es
el mismo problema de siempre, más chico: acotar más la región, o cortar por
profundidad, lo seguiría reduciendo.

## Cómo se sabe que las cuentas están bien

Cada paso se contrastó contra un caso de respuesta conocida:

| Qué | Control | Resultado |
|---|---|---|
| Vecino más cercano | Bucle O(N²) ingenuo sobre 300 eventos | 0 padres distintos |
| Vecino más cercano | Mismo cálculo con bloques de 7 y de 256 | Resultado idéntico |
| Bimodalidad | Catálogo sintético **sin** réplicas plantadas | Unimodal: no inventa un segundo modo |
| Bimodalidad | Catálogo sintético **con** 50% de réplicas plantadas | Detecta 48.4%, separación 4.72 desvíos |
| Mezcla gaussiana | Mezcla sintética de parámetros conocidos | Recupera pesos, medias y desvíos con dos decimales |
| Thinning | Catálogo con 50% de réplicas plantadas | Marca 49.6%, precisión 94.6%, recall 93.8% |
| Thinning | Catálogo **sin** réplicas plantadas | Marca 25.6% — pero el contraste con el nulo da 0.0 puntos de exceso y avisa que eso es azar |
| Semilla | Cinco semillas sobre el catálogo real | Entre 375 y 390 réplicas (42.8–44.5%); con la misma semilla, idéntico |
| Clusters | Bosque armado a mano con cadena A→B→D y un premonitor | Topología exacta: D hereda el cluster de A a través de B, y el premonitor no queda como sismo principal |
| Clusters | Invariantes sobre el catálogo real | Los eventos por cluster suman el total, cada evento cae en exactamente un cluster, y réplicas + premonitores + 1 = tamaño |
| Productividad | Catálogos sintéticos con α conocido | Recupera 0.7 → 0.699, 0.9 → 0.904, 1.1 → 1.105 |

## Lo que falta

Las tareas del método están todas, y el catálogo ya se consulta acotado a un
rectángulo. Lo que queda es afinar ese recorte y dos guardas que se quedaron
cortas:

| Pendiente | Qué pasa |
|---|---|
| Corte por profundidad | En Argentina importa tanto como el rectángulo: la sismicidad superficial andina y la del slab profundo (100–250 km) son poblaciones distintas con estadística distinta, y mezclarlas ensucia `d` y eta. El FDSN acepta `mindepth`/`maxdepth`. |
| El contraste con el nulo usa puntos absolutos | Argentina dio 9.8% contra 4.8% — un exceso de **2×**, que es señal — pero como en puntos absolutos son 5.0 y el umbral es 5.0, avisó igual. La corrida global dio 41.9 contra 37.9, que es 1.1× y no significa nada, y avisó lo mismo. Debería ser un cociente. |

## Referencias

- Zaliapin, Gabrielov, Keilis-Borok & Wong (2008), *Clustering analysis of
  seismicity and aftershock identification* — la métrica eta.
- Zaliapin & Ben-Zion (2013), *Earthquake clusters in southern California I* —
  el thinning y la reconstrucción de clusters.
- Aki (1965) — estimación de `b` por máxima verosimilitud.
- Wiemer & Wyss (2000) — Mc por bondad de ajuste.
- Grassberger & Procaccia (1983) — dimensión de correlación.
