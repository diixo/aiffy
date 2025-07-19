import random
import json

# --- Примеры сущностей ---
PERSONS = [
    "Elon Musk", "Jeff Bezos", "J.K. Rowling", "Barack Obama",
    "Albert Einstein", "Mark Zuckerberg", "Steve Jobs", "Bill Gates"
]

ORGS = [
    "SpaceX", "Amazon", "Apple", "Microsoft", "Facebook",
    "Tesla", "Google", "OpenAI"
]

GPES = [
    "the United States", "Germany", "Canada", "France", "Japan",
    "the United Kingdom", "Australia"
]

WORKS = [
    "Harry Potter", "The Lord of the Rings", "Pride and Prejudice",
    "A Brief History of Time", "The Hobbit", "Novel-1984"
]

# --- Сколько примеров ---
N = 200

# --- Шаблоны ---
templates = [
    {
        "source": "Fill-in: [PERSON] founded [ORG] in [GPE].",
        "pattern": "{person} founded {org} in {gpe}."
    },
    {
        "source": "Fill-in: [PERSON] wrote [WORK].",
        "pattern": "{person} wrote {work}."
    },
    {
        "source": "Fill-in: [PERSON] was born in [GPE].",
        "pattern": "{person} was born in {gpe}."
    }
]

# --- Генерация ---
data = []

for _ in range(N):
    tmpl = random.choice(templates)
    target = tmpl["pattern"].format(
        person=random.choice(PERSONS),
        org=random.choice(ORGS),
        gpe=random.choice(GPES),
        work=random.choice(WORKS)
    )
    example = {
        "source": tmpl["source"],
        "target": target
    }
    data.append(example)


# --- Сохраняем в JSONL ---
# with open("mask_fill_corpus.jsonl", "w", encoding="utf-8") as f:
#     for ex in data:
#         f.write(json.dumps(ex) + "\n")

print(f"✅ Сгенерировано {N} примеров в mask_fill_corpus.jsonl")


#################################################################

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm

# --------- Настройки ----------
MODEL_NAME = "gpt2"  # или "gpt2-medium", "gpt2-xl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 3e-5


examples = data

# --------- Токенизация ----------
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

# GPT обычно не имеет BOS/EOS — добавим
tokenizer.pad_token = tokenizer.eos_token

# --------- Кастомный Dataset ----------
class MaskFillDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Полный input: prompt + правильный ответ
        full_input = ex['source'] + " " + ex['target']
        encoding = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Для GPT-2 мы просто сдвигаем labels == input_ids
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

dataset = MaskFillDataset(examples, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------- Оптимизатор ----------
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# --------- Тренировка ----------
model.train()

for epoch in range(EPOCHS):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

print("✅ Training done!")
#model.save_pretrained("./gpt2_mask_fill_finetuned")
#tokenizer.save_pretrained("./gpt2_mask_fill_finetuned")

##############################################################

model.eval()  # режим инференса

prompt = "Fill-in: [PERSON] founded [ORG] in [GPE]."
model.to("cuda" if torch.cuda.is_available() else "cpu")


prompts = [
    "Fill-in: [PERSON] founded [ORG] in [GPE].",
    "Fill-in: [PERSON] was born in [GPE].",
    "Fill-in: [PERSON] wrote [WORK].",
]

inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=30,   # сколько максимум токенов в ответе
    num_return_sequences=1,
    do_sample=True,  # для разнообразия
    top_p=0.95,      # nucleus sampling
    top_k=10         # или можно top_k=50
)
#     num_beams=3,       # beam search
#     do_sample=False,   # beam делает лучше для фактов
#     num_return_sequences=1
# )

for i, out in enumerate(output):
    full_text = tokenizer.decode(out, skip_special_tokens=True)
    # Убираем prompt из результата, чтобы видеть только дополнение
    predicted = full_text[len(prompts[i]):].strip()
    print(f"\n🔹 PROMPT: {prompts[i]}\n🔹 OUTPUT: {predicted}")

# decoded = tokenizer.decode(output[0], skip_special_tokens=True)
# print(decoded)
