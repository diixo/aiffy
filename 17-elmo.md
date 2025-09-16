
# ELMo v3 (https://tfhub.dev/google/elmo/3)

### 1. signature="default"

Вход - **целые предложения строкой** `(batch,)`.

* Токенизацию делает сам ELMo внутри.

* **На выходе** - pooled embedding `(batch, 1024)`.

### 2. signature="tokens"
Вход — **список токенов на уровне слов**, паддинги и sequence_len подаёшь вручную.

* **На выходе** - `(batch, seq_len, 1024)` для каждого токена.
