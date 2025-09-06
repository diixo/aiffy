
import torch
from transformers import AutoModelForCausalLM
import tiktoken


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
model = model.eval()

enc = tiktoken.get_encoding("gpt2")

pad_token_id = 50256   # <|endoftext|> = eos = pad


def pad_or_truncate(tokens, max_len=1024, pad_token_id=50256):
    tokens = [enc.decode([tid]) for tid in tokens]
    print(tokens)
    if len(tokens) > max_len:
        return tokens[-max_len:]  # обрезаем слева
    return tokens + [pad_token_id] * (max_len - len(tokens))


prompts = [
    "Once upon a time",
    "The meaning of life is",
    "In the future, AI will",
    "Python is a programming language",
    "The capital of France is",
    "To be or not to be",
    "Maybe yes, maybe no",
    "instruction: Explain the process of erosion. Erosion is the process by which soil, rock, and other surface material are worn away and transported by natural forces such as wind or water. It can shape landscapes and affect ecosystems."
]

max_len = 128

batch = [pad_or_truncate(enc.encode(p), max_len, pad_token_id) for p in prompts]

real_max_len = max(len(idx)+1 for idx in batch)
print("real_max_len:", real_max_len, len(prompts))

input_ids = torch.tensor(batch, device=device)

# 4. Автогенеарция (простая версия, жадная)
max_new_tokens = 50
for _ in range(max_new_tokens):
    with torch.no_grad():
        print("shape:", input_ids.shape)
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # берём только последний шаг
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_id], dim=-1)

# 5. Декодинг обратно в текст
generated_text = enc.decode(input_ids[0].tolist())
print(generated_text)
