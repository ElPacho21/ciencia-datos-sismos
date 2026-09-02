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

### `refine_silver` → el catálogo limpio

Parsea fechas y números, tira las filas sin magnitud o sin epicentro, saca
duplicados y **ordena por tiempo**. Nada de método todavía, sólo higiene.

Está separado a propósito: silver no depende de ningún parámetro del algoritmo,
así que se calcula una vez y se reutiliza aunque después cambies el método de
Mc o el umbral. El orden cronológico es parte del contrato — el vecino más
cercano recorre el catálogo hacia atrás y lo da por sentado.

### `estimate_mc_b_d` → los tres números que el método necesita

Acá está la parte que más se olvida: **`b` y `d` no son constantes de tabla**,
son parámetros que hay que ajustar a *estos* datos. Y antes de poder ajustarlos
hay que resolver un problema previo.

**Mc, la magnitud de completitud.** Un catálogo registra todos los sismos
grandes y se le escapan los chicos: no hay sismógrafos en todos lados. Mc es la
magnitud a partir de la cual ya no se le escapa nada. Todo lo que se calcule
por debajo de Mc sale sesgado, porque faltan eventos. Por eso Mc va primero y
el catálogo se recorta ahí antes de seguir.

**`b`, la pendiente de Gutenberg-Richter.** Por cada sismo de magnitud 6 hay
como diez de magnitud 5 y cien de magnitud 4. Esa proporción es `b`, y en la
práctica da cerca de 1 en casi todo el mundo. Sirve como control: si `b` da
muy lejos de 1, casi siempre es que Mc quedó mal.

**`d`, la dimensión fractal.** Mide cuán apelotonados están los epicentros. Se
calcula contando, para cada radio, qué fracción de todos los pares de sismos
está más cerca que eso. Si estuvieran repartidos parejo por una superficie esa
fracción crecería como `r²`; si estuvieran alineados sobre una falla, como `r¹`.
Lo que da en el medio es `d`.

Los tres salen a un JSON que guarda además las **curvas diagnósticas** (la
distribución de magnitudes y la integral de correlación), para poder graficarlas
y defender los números sin recalcular la parte cara.

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

## Una corrida de referencia

Catálogo global, del 1 al 30 de agosto de 2026, magnitud mínima 2.5:

| | |
|---|---|
| Eventos en el catálogo limpio | 2169 |
| Mc (bondad de ajuste) | 4.30, explica el 90.6% de la distribución |
| `b` | 0.956 ± 0.028, sobre 876 eventos completos |
| `d` | 1.756 (R² 0.9919, entre 0.7 y 65 km) |
| Umbral | log₁₀ η₀ = −4.637 |
| Por debajo del umbral | 367 de 875 (41.9%) |
| Réplicas después del thinning | 375 de 876 (42.8%), semilla 1 |

Tres cosas que vale la pena mirar de esa corrida:

**El contraste entre los dos métodos de Mc.** Con máxima curvatura, Mc baja a
2.80 y `b` se desploma a **0.332**, que es físicamente imposible. Es la firma
exacta de un Mc subestimado: al catálogo global le faltan los eventos chicos, la
distribución se aplana y la pendiente se va al piso. Por eso el default es
bondad de ajuste, y por eso el código avisa cuando `b` sale fuera de 0.5–2.0.

**La bimodalidad está, pero justa.** Los dos modos quedan separados 2.82
desvíos y el modelo estima un 6.7% de error de clasificación. Pasa el piso, pero
sin holgura: el valle es más un hombro que un pozo.

**El contraste con el catálogo barajado dispara la alarma.** Bajo el umbral cae
el 41.9% del catálogo real y el 37.9% del barajado: un exceso sobre el azar de
apenas 4.1 puntos, contra los 8.0 que da un control sintético con réplicas
plantadas de verdad. El contraste es *conservador* por construcción —el nulo
conserva N y por lo tanto es más denso que el fondo verdadero, lo que infla su
proporción bajo el umbral— así que un exceso chico no prueba que no haya
réplicas. Pero sí dice que sobre un mes de catálogo global la señal es débil, y
es el argumento más fuerte para acotar la región.

## Cómo se sabe que las cuentas están bien

Cada paso se contrastó contra un caso de respuesta conocida:

| Qué | Control | Resultado |
|---|---|---|
| `b` | Catálogo sintético con `b` conocido | Lo recupera con 0.3–1% de error; sin la corrección de binning se va 12.6% arriba |
| `d` | Nube uniforme en 2D, donde `d` tiene que dar 2 | 1.906 — el sesgo bajo del ~5% es propio del método, no del código |
| Vecino más cercano | Bucle O(N²) ingenuo sobre 300 eventos | 0 padres distintos |
| Vecino más cercano | Mismo cálculo con bloques de 7 y de 256 | Resultado idéntico |
| Bimodalidad | Catálogo sintético **sin** réplicas plantadas | Unimodal: no inventa un segundo modo |
| Bimodalidad | Catálogo sintético **con** 50% de réplicas plantadas | Detecta 48.4%, separación 4.72 desvíos |
| Mezcla gaussiana | Mezcla sintética de parámetros conocidos | Recupera pesos, medias y desvíos con dos decimales |
| Thinning | Catálogo con 50% de réplicas plantadas | Marca 49.6%, precisión 94.6%, recall 93.8% |
| Thinning | Catálogo **sin** réplicas plantadas | Marca 25.6% — pero el contraste con el nulo da 0.0 puntos de exceso y avisa que eso es azar |
| Semilla | Cinco semillas sobre el catálogo real | Entre 375 y 390 réplicas (42.8–44.5%); con la misma semilla, idéntico |

## Lo que falta

| Tarea | Qué haría |
|---|---|
| `build_clusters` | Recorrer el bosque. Agrupar por `parent_id` cuenta sólo los hijos directos, y una réplica también tiene réplicas: el sismo principal es la raíz del árbol entero. Recién ahí sale la cuenta de réplicas por sismo principal. |

Y un pendiente que no es una tarea sino una decisión de método: **acotar el
catálogo a una región**. Es el cambio que más mejoraría todos los números de
acá. Un catálogo global mezcla zonas con completitudes muy distintas, y le da a
`d` una geometría que responde a los bordes de placa y no a una ley de
potencias.

## Referencias

- Zaliapin, Gabrielov, Keilis-Borok & Wong (2008), *Clustering analysis of
  seismicity and aftershock identification* — la métrica eta.
- Zaliapin & Ben-Zion (2013), *Earthquake clusters in southern California I* —
  el thinning y la reconstrucción de clusters.
- Aki (1965) — estimación de `b` por máxima verosimilitud.
- Wiemer & Wyss (2000) — Mc por bondad de ajuste.
- Grassberger & Procaccia (1983) — dimensión de correlación.
