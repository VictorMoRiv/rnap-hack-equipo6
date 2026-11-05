# rnap-hack-equipo6
This is a project for RNAP Hack 2026
#ENG

#ESP
Pista 5 — La interfaz que sigue tu intención 

Acerca de la empresa. Casa Norte construye un asistente de hogar en español para familias del norte de México. (Empresa ficticia.) 

El problema hoy. El asistente siempre ha respondido con mensajes de texto. Los usuarios le han pedido que configure alarmas y agregue artículos a la lista de compras, y les ha respondido con largas frases que deben leer y poner en práctica en otra aplicación. En las pruebas de usuario, si el asistente hubiera mostrado una tarjeta sencilla con un botón, la gente habría terminado la tarea en la mitad del tiempo. 

El gran desafío. ¿Puede su sistema comprender lo que el usuario quiere de una oración, seleccionar los datos clave (hora, lugar, persona, elemento) y mostrar el componente de pantalla que completa la tarea? 

Posibles enfoques. 

Línea base de intención. TF-IDF con un clasificador clásico, o un LSTM pequeño, para elegir la intención. 

Transformador afinado. Ajuste un BERT español o un DistilBERT multilingüe según las intenciones que elija. 

Datos clave. Clasificación de tokens con las etiquetas de ranura del conjunto de datos, o un modelo de lenguaje que devuelve JSON con los campos. 

Componentes que reaccionan. En Gradio, cada intención muestra su propia tarjeta (alarma, clima, lista de compras, calendario) y oculta el resto. Cuando el usuario cambia un valor en la tarjeta, el estado de la aplicación también cambia. 

Estas son instrucciones, no requisitos. Un flujo completo obtiene una puntuación superior a cinco pantallas a medio terminar. 

Recursos. MASSIVE, locale es-ES, en huggingface.co/datasets/AmazonScience/massive, con 60 intenciones y espacios etiquetados. Elige de 6 a 8 intenciones y entrena solo con ellas. Mida la precisión de la intención y F1 en el conjunto de prueba, y el porcentaje de datos clave extraídos correctamente. Agrega 30 oraciones al estilo mexicano escritas por tu equipo como prueba extra. 

Qué construir. Una interfaz donde el usuario escribe una oración. La tarjeta coincidente aparece con sus datos ya completados y al menos una acción en la tarjeta produce un cambio real, como una alarma agregada a la lista de alarmas. 

Prueba del juez. El juez escribe frases informales que tu equipo no ha visto, como “despiértame a las seis y cuarto”. 

Por qué es importante. Si los asistentes crearan la pantalla adecuada para cada solicitud, las personas con poca experiencia con aplicaciones completarían las tareas por su cuenta. 

Criterios de evaluación. 

Resultados: intención correcta y datos correctos sobre las sentencias del juez. 

Interfaz: la tarjeta cambia con la intención y su acción funciona. 

Robustez: español mexicano informal y frases fuera de las intenciones elegidas. 

Claridad: el usuario entiende lo que entendió el asistente. 
