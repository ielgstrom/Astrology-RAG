# FutureReader

Una vidente en la terminal. Preguntale por tu horóscopo o por lo que te espera y
te responderá como toda buena adivina: en tono misterioso y sin concretar
demasiado.

Funciona con un modelo [Ollama](https://ollama.com) local: nada sale de tu
máquina y no hace falta ninguna API key.

## Instalación

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama serve            # si no está ya arrancado
ollama pull llama3.2    # o el modelo que prefieras
```

## Cómo usarlo

```sh
python FutureReader.py
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
| `--model` | modelo de Ollama (por defecto `llama3.2`) |
| `--num-ctx` | tamaño del contexto en tokens (por defecto 8192); súbelo si cargas fuentes grandes, porque Ollama recorta en silencio lo que no cabe — y cuando recorta, la vidente empieza a nombrar cartas que nunca salieron |
| `--num-predict` | longitud máxima de la respuesta (`-1` = sin límite) |
| `--temperature` | temperatura de la respuesta; alto = más ambiguo |
| `--base-url` | servidor de Ollama remoto (por defecto `$OLLAMA_HOST`) |
| `--no-stream` | espera la respuesta entera en vez de irla escribiendo |
| `--verbose` | Muestra mensajes de consumo de tokens de la sesión|