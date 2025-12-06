mbed
===

A CLI for indexing directories for vector search built on `llama-index`.

install for dev
---

```
git clone ...
cd ...
uv sync
```

use
---

**index a directory**
```
[uv run] mbed init [-d DIRECTORY] 
```

**search within an indexed directory**
```
[uv run] mbed search [-d DIRECTORY] "QUERY"
```

**update the index of a directory for any changes since last index**
```
[uv run] mbed update [-d DIRECTORY]
```

testing
---

For quick higher-level tests

```bash
just test
```

To run the full integration test with real embeddings (slower):

```bash
just test-embedding
```

