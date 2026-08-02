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
A partir de aqui te conducirá a un chat con una vidente virtual. Para acabar con el chat puedes escribir:

```
quit
```

Como librería:

## Cómo funciona

- `SYSTEM_PROMPT` en [FutureReader.py](FutureReader.py) le da el papel: solo
  habla del futuro. Si le preguntas por el pasado o el presente, contesta
  *"The future is my concert, thus I cannot answer that question."*
- Las fuentes que cargues (por ejemplo
  [astrology-sign-meanings.pdf](astrology-sign-meanings.pdf), con el
  significado de cada signo) se meten enteras en el prompt para que las
  respuestas tengan algo de base astrológica. Sin vector store ni chunking.
- Formatos admitidos: texto plano y markdown, `.pdf`, `.docx`, URLs y
  directorios completos.

## Opciones

| Opción | Para qué sirve |
| --- | --- |
| `--model` | modelo de Ollama (por defecto `llama3.2`) |
| `--num-ctx` | tamaño del contexto en tokens (por defecto 2048); súbelo si cargas fuentes grandes, porque Ollama recorta en silencio lo que no cabe |
| `--num-predict` | longitud máxima de la respuesta (`-1` = sin límite) |
| `--temperature` | temperatura de la respuesta; alto = más ambiguo |
| `--base-url` | servidor de Ollama remoto (por defecto `$OLLAMA_HOST`) |
| `--no-stream` | espera la respuesta entera en vez de irla escribiendo |
| `--verbose` | Muestra mensajes de consumo de tokens de la sesión|