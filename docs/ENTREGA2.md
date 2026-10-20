
En la Entrega 1 la pregunta era si tenías un pipeline que produce tu dataset sin que nadie lo toque a mano. Ahora que el dataset existe, la pregunta es otra: ¿sabés qué tiene adentro, y podés defender con un número cada decisión que tomaste sobre él?

Es la entrega que decide con qué llegás a modelar. Todo lo que salga de acá — qué columnas entran, cuál es el objetivo, qué hay que transformar — es la materia prima de la Entrega 3.

Qué tenés que traer
Llegá con esto ya hecho. Los 20 minutos son para mostrar y explicar, no para correr celdas ni para esperar que termine un gráfico.

Qué	Detalle
El notebook ejecutado	De arriba a abajo, sin errores y con las salidas guardadas. Abierto antes de entrar. Si hay que correrlo delante nuestro, son minutos que no recuperás.
El perfil del dataset	Filas, columnas, tipos, nulos, constantes, duplicados y la distribución de la variable objetivo. Los números a mano (ver más abajo).
Cuatro fichas de hipótesis	Con los seis campos, el número, la zona del semáforo y el gráfico.
La tabla de columnas candidatas	Qué columnas le vas a dar al modelo, cuáles descartás y por qué.
Lo que te marcamos en la Entrega 1	Resuelto, o con una explicación de por qué no se pudo.
Cómo se organizan los 20 minutos
Min.	Momento	Qué pasa
0–2	Qué cambió	Qué se corrigió del pipeline desde la Entrega 1, y si cambió algo de la pregunta del proyecto. Dos oraciones.
2–7	El perfil	Recorrés el dataset: cuántas filas, cuál es la clave, dónde hay nulos y por qué, cuál es la columna objetivo y cómo está distribuida.
7–13	Dos hipótesis	El docente elige dos de tus cuatro fichas y las recorrés enteras, desde la afirmación hasta la decisión.
13–18	Preguntas dirigidas	El docente pregunta, a integrantes puntuales, sobre cualquier parte de lo anterior.
18–20	Devolución	Qué está bien, qué hay que corregir antes de la Entrega 3.
La evaluación es individual. Repartirse el trabajo está bien; desentenderse de la parte del otro, no. La pregunta sobre una hipótesis puede caerle a quien hizo la limpieza.

Lo que se verifica con un número
Esto lo podés medir vos mismo antes de venir. Si tu dataset está en un DataFrame llamado df:

Qué	Cómo lo verificás	Qué tenés que saber decir
Tamaño	df.shape	Cuántas filas y columnas, y qué es una fila.
Tipos	df.dtypes.value_counts()	Si alguna numérica quedó como texto, y por qué.
Nulos por columna	df.isna().mean().sort_values(ascending=False)	De dónde salen los faltantes de las tres columnas peores.
Columnas sin información	df.columns[df.nunique() <= 1]	Cuáles son constantes y si se van.
Duplicados de la clave	df["tu_clave"].duplicated().sum()	Que dé cero, o por qué no da cero.
Distribución del objetivo	df["objetivo"].value_counts(normalize=True)	Si hay una clase que se come todo.
Asimetría de las numéricas	df.select_dtypes("number").skew().sort_values()	Cuáles tienen cola larga y habría que transformar.
La distribución del objetivo es el número más importante de esta entrega, y el que más se pasa por alto. Si el 92 % de tus filas son de una sola clase, el modelo de la Unidad 3 arranca con un problema que conviene ver ahora y no en octubre.

Las cuatro hipótesis
Las mismas seis casillas del TP2, sobre tu dataset. Cuatro fichas, con estas condiciones:

Al menos una de cada pregunta: una que responda algo sobre tu dominio y una que decida sobre una columna del modelo.
Al menos una refutada o inconclusa. Cuatro hipótesis confirmadas suele querer decir que se eligieron cosas que ya se sabían.
Cada resultado con su número y su zona, no sólo con un gráfico.
Los movimientos justificados por lo que se ve en el gráfico, y ninguna hipótesis con más de dos.
La tabla de columnas candidatas
Es el entregable que se lleva la Entrega 3. Una fila por columna que estés pensando darle al modelo:

Columna	Qué mide	Zona	Decisión	¿Existiría al predecir?
ejemplo_a	Antigüedad del cliente en meses	verde	Entra	Sí
ejemplo_b	Motivo de la baja	verde	Sale	No — se completa después del hecho
ejemplo_c	Monto facturado	amarillo	Entra en escala logarítmica	Sí
La última columna es el chequeo de fuga, y es la que más caro se paga si falta. Antes de darle una columna al modelo, preguntate si esa columna existiría en el momento de predecir. Si la respuesta ya está adentro, no es una feature: es la respuesta escrita de otra forma, y el modelo va a dar buenísimo por el peor de los motivos.

Lo que hay que saber explicar
Esto no lo mide ningún script: se pregunta.

Cuál es la variable objetivo y cómo está distribuida. Señalá una columna concreta, no una idea.
Por qué hay nulos donde hay nulos. No pedimos cero nulos: eso no existe en datos reales. Distinguí los faltantes de verdad de los nulos estructurales, que son los que tienen que estar vacíos.
Qué hipótesis te salió mal, y qué hiciste con eso.
Qué columna descartaste por fuga, y en qué momento se completa en la realidad.
Qué quedó inconcluso y qué haría falta para resolverlo.
Cómo se aprueba
El notebook corre entero y las salidas están guardadas.
El grupo puede describir su dataset con números, no con adjetivos.
Las cuatro fichas están completas y cada una termina en una decisión.
La tabla de columnas candidatas existe y cada fila pasó el chequeo de fuga.
Las respuestas no vienen de una sola persona.
Tres formas de perder los 20 minutos
El EDA de checklist. Veinte gráficos, ninguna conclusión. Un histograma por columna no es un análisis exploratorio: es un catálogo. Lo que se corrige es qué decidiste, no cuántas celdas corriste.
Contestar con el gráfico. «Se ve que hay diferencia» no es un resultado. Un gráfico no se compara contra nada; el número sí, y por eso es el que decide.
Que hable uno solo. Es el error más caro, porque las preguntas van dirigidas y la nota es individual.
Si algo no te dio, vení igual y decilo. Un grupo que muestra un análisis a medias y sabe dónde está trabado se lleva una devolución útil. Uno que no se presenta, no.

