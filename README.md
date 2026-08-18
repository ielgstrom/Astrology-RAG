# FutureReader

Una vidente en la terminal. Preguntale por tu horóscopo o por lo que te espera y
te responderá como toda buena adivina: en tono misterioso y sin concretar
demasiado.

Funciona con dos motores, y elige solo:

- **[Mistral](https://mistral.ai)** (`mistral-small-latest`): hace falta una API
  key, pero el modelo es mucho más grande y sigue mejor el hilo de la lectura.
  Se usa por defecto **si hay una key en el entorno**.
- **[Ollama](https://ollama.com)** local (`llama3.2`): nada sale de tu máquina y
  no hace falta ninguna key. Es a lo que recurre cuando no hay key.

Con `--provider mistral` o `--provider ollama` fuerzas uno u otro.

## Instalación

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Para usar Mistral, copia `.env.example` a `.env` y pon tu key
([console.mistral.ai](https://console.mistral.ai/api-keys)):

```sh
MISTRAL_API_KEY=...
```

Para usar el modelo local en su lugar:

```sh
ollama serve            # si no está ya arrancado
ollama pull llama3.2    # o el modelo que prefieras
```

## Cómo usarlo

```sh
python cli.py
```
Lo primero que te preguntará es qué lectura quieres:

```
  1) A spread of three cards
  2) A reading of your sign
```

Con **1** reparte tres cartas (pasado / presente / futuro) y construye la lectura
sobre ellas. Con **2** lee lo que dicen los astros para tu signo. En ambos casos
te pedirá la fecha de nacimiento, porque toda profecía se lee a través del signo.

Después te conducirá a un chat con la vidente, donde puedes seguir preguntando.
Para acabar con el chat puedes escribir:

```
quit
```

## Opciones

| Opción | Para qué sirve |
| --- | --- |
| `--provider` | `auto` (por defecto), `mistral` u `ollama`. `auto` = Mistral si hay key, Ollama si no |
| `--model` | modelo a usar (por defecto `mistral-small-latest` en Mistral, `llama3.2` en Ollama) |
| `--num-ctx` | tamaño del contexto en tokens (8192 en Ollama, 32768 en Mistral); súbelo si cargas fuentes grandes, porque Ollama recorta en silencio lo que no cabe — y cuando recorta, la vidente empieza a nombrar cartas que nunca salieron. En Mistral la ventana la pone el servidor, así que este número solo sirve para el aviso de consumo |
| `--num-predict` | longitud máxima de la respuesta (`-1` = sin límite) |
| `--temperature` | temperatura de la respuesta; alto = más ambiguo. Mistral no admite más de `1.0` y recorta ahí |
| `--base-url` | servidor de Ollama remoto (por defecto `$OLLAMA_HOST`); no aplica a Mistral |
| `--no-stream` | espera la respuesta entera en vez de irla escribiendo |
| `--verbose` | Muestra mensajes de consumo de tokens de la sesión|

## La misma vidente por HTTP

La lectura vive en `FutureReader.py` y no sabe nada de dónde se lee. `cli.py` la
saca por la terminal; `api.py` la sirve por HTTP para que la lea un navegador:

```sh
uvicorn api:app --reload
```

| Endpoint | Qué hace |
| --- | --- |
| `GET /health` | qué motor respondería ahora mismo |
| `POST /sessions` | abre una lectura (`reading`, `topic`, `birth_date`) y devuelve el `session_id`, el signo y las cartas repartidas |
| `GET /sessions/{id}/reading` | la lectura entera en streaming (SSE) |
| `GET /sessions/{id}/ask?q=…` | una pregunta libre, también en streaming |

Los dos endpoints de streaming hablan **Server-Sent Events**, que en el navegador
se leen con `EventSource`. Cada trama va nombrada: `card` cuando se voltea una
carta, `chunk` por cada trozo de texto, `answer_end` al acabar una respuesta,
`end` al acabar la lectura y `error` si el motor falla.

Las sesiones viven en memoria: un reinicio se las lleva todas y ninguna caduca.

Los docs interactivos quedan en `http://127.0.0.1:8000/docs`.