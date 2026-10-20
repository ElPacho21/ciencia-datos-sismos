Esta entrega tiene una sola pregunta de fondo: ¿tenés un pipeline que produce tu dataset sin que nadie lo toque a mano? Todo lo demás son formas de responderla.

Qué tenés que traer
Llegá con esto ya funcionando. Los 20 minutos son para mostrar y explicar, no para instalar ni para esperar que termine una descarga.

Qué	Detalle
El entorno levantado	Airflow corriendo en la notebook de alguien del grupo, con la interfaz abierta en el navegador. Prendelo antes de entrar.
Una corrida en verde	Una ejecución completa y exitosa, ya hecha, visible en el historial. No la corras en el momento: si tu pipeline tarda cinco minutos, son cinco minutos de los veinte.
El CSV producido	El archivo que generó esa corrida, abierto o listo para abrir.
El chequeo de calidad	Los siete criterios verificados sobre tu salida, con los números a mano (ver más abajo).
Cómo se organizan los 20 minutos
Min.	Momento	Qué pasa
0–2	De qué se trata	La pregunta del proyecto en una frase, la fuente, y qué es una fila de tu dataset. Tres oraciones, sin filminas.
2–7	La corrida	Mostrás la ejecución en verde en la interfaz y recorrés el grafo explicando qué hace cada tarea y por qué está donde está.
7–12	El dataset	Abrís el CSV y lo defendés contra los siete criterios: cuántas filas, cuál es la clave, dónde hay nulos, cuál es la columna objetivo.
12–18	Preguntas dirigidas	El docente pregunta, a integrantes puntuales, sobre cualquier parte de lo anterior.
18–20	Devolución	Qué está bien, qué hay que corregir antes de la Entrega 2.
La evaluación es individual. Repartirse el trabajo está bien; desentenderse de la parte del otro, no. La pregunta sobre el scraping puede caerle a quien escribió la transformación. Con grupos de cuatro a siete personas esto pesa: no alcanza con que uno solo sepa todo.

Qué es un buen dataset de salida
Los siete criterios que usaste en la primera clase para elegir una fuente son los mismos que ahora usamos para juzgar lo que produjo tu pipeline. Cambió una sola cosa, y no es menor: en agosto la calidad del dato era responsabilidad de quien publicó la fuente. Ahora es tuya.

Lo que se verifica con un número
Estos los podés medir vos mismo antes de venir. Si tu CSV está cargado en un DataFrame llamado df:

Criterio	Qué tiene que dar	Cómo lo verificás
Clave sin duplicados	True	df["tu_clave"].is_unique
Volumen suficiente	más de 1.000 filas	len(df)
Ancho suficiente	5 o más columnas útiles	df.shape
Mezcla de tipos	numéricas, categóricas y fechas	df.dtypes.value_counts()
Nulos conocidos	saber cuáles y por qué	df.isna().mean().sort_values(ascending=False)
Sin columnas vacías	ninguna al 100 % de nulos	df.columns[df.isna().all()]
La clave primaria es el criterio más importante, y el que más se subestima. Es el test operativo de la unidad de análisis: si la clave repite, o la unidad está mal definida o el pipeline está duplicando filas. No existe un dataset con la clave limpia y la unidad equivocada.

Lo que hay que saber explicar
Esto no lo mide ningún script: se pregunta.

«Una fila es un ___». Completá la frase en una oración, y que coincida con la pregunta del proyecto. Si tu pregunta es sobre pacientes y una fila es una consulta médica, hay un problema.
Cuál es la columna objetivo, o qué estructura latente esperás encontrar. Señalá una columna concreta del CSV, no una idea.
Por qué hay nulos donde hay nulos. No pedimos cero nulos: eso no existe en datos reales. Pedimos que sepas cuáles tenés y qué significan.
Qué pasa si lo corrés de nuevo. ¿Sale el mismo archivo? Si no, ¿por qué?
Dónde guardás el dato crudo. ¿Queda en algún lado la respuesta de la API, el HTML o el archivo original, tal como llegó? Si mañana descubrís que venías interpretando mal una columna, ¿podés arreglarlo sin volver a la fuente? El dataset que entregás es la capa plata; la pregunta es si además tenés el bronce.
Sobre «tidy», y una confesión
Tidy significa: cada fila una observación, cada columna una variable. Suena simple hasta que lo aplicás a datos reales.

El dataset canónico de la cátedra no cumple del todo esa regla. La columna player_positions guarda más de un valor en la misma celda —«CM, CDM»— en el 65 % de las filas, y player_traits en el 24 %.

No es un descuido: es una concesión deliberada. Separarlo en columnas o en filas complicaría todo lo demás, y por eso el dataset trae además best_position, que sí tiene un solo valor y es la que se usa como objetivo en la Unidad 3.

Si tu dataset tiene una concesión parecida, no la escondas: explicala. Un grupo que sabe dónde su dataset se aparta del ideal y por qué está mejor parado que uno que repite la definición de memoria.

Cómo se aprueba
El piso es el mismo para todos, más allá de si tu fuente es una API, un CSV que se descarga o una página que hay que scrapear:

Hay una corrida completa en verde, hecha por ustedes, visible en la interfaz.
El grupo sabe explicar qué hace cada tarea del pipeline y por qué está ahí.
El pipeline produce un CSV que cumple los criterios medibles.
El grupo sabe defender las decisiones de la lista de arriba.
Las respuestas no vienen de una sola persona.
No hacer esto
Llegar sin el entorno levantado. Arrancar Airflow delante nuestro se lleva la mitad del tiempo, y es tiempo que no recuperás.
Correr el pipeline en vivo. Traé la corrida hecha. Si querés mostrar una ejecución en el momento, que sea sobre una muestra chica y además de la que ya está.
Que hable uno solo. Es el error más caro, porque las preguntas van dirigidas y la nota es individual.
Si algo de tu pipeline no llegó a funcionar, vení igual y decilo. Un grupo que muestra un pipeline a medias y entiende dónde está trabado se lleva una devolución útil. Uno que no se presenta, no.