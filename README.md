# FutureReader

A small LangChain wrapper around a local [Ollama](https://ollama.com) model that
answers questions about files, directories, and web pages. Nothing leaves the
machine and there is no API key.

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama serve            # if it is not already running
ollama pull llama3.2    # or whatever model you want to use
```

## Use

```sh
# one question
python FutureReader.py notes.md report.pdf -q "What are the open risks?"

# interactive, with follow-up questions that remember the conversation
python FutureReader.py ./docs https://example.com/spec
```

As a library:

```python
from FutureReader import FutureReader

reader = FutureReader()
reader.add("notes.md", "./docs", "https://example.com/spec")
print(reader.ask("Summarise the three main themes."))
```

## What it does

Sources are loaded into `Document`s and stuffed whole into the prompt — no
vector store, no chunking. That works as long as everything fits in the model's
context window (`--num-ctx`); past that, add retrieval.

| Source | Handled by |
| --- | --- |
| `.txt`, `.md`, code, config, any UTF-8 text | direct read |
| `.pdf` | `PyPDFLoader` (needs `pypdf`) |
| `.docx` | `Docx2txtLoader` (needs `docx2txt`) |
| `http(s)://` | `WebBaseLoader` (needs `beautifulsoup4`) |
| a directory | walked recursively, skipping hidden dirs, `node_modules`, `.venv`, etc. |

## Notes

- Default model is `llama3.2`. Override with `--model` (anything in
  `ollama list` works).
- **`--num-ctx` is the setting that matters.** Ollama's own default is 2048
  tokens and it silently drops everything past the limit, so the tool defaults
  to 16384 and warns when the loaded sources come close to filling it. Raise it
  for large source sets, lower it if you run out of RAM.
- `--num-predict` caps the answer length (`-1` for unlimited);
  `--temperature` defaults to 0 for grounded answers.
- `--base-url` points at a remote Ollama server; it defaults to `$OLLAMA_HOST`,
  then `http://localhost:11434`.
- Small models cite less reliably than large ones. If citations get sloppy, try
  a bigger model before rewriting `SYSTEM_PROMPT`.
