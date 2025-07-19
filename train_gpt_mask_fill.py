
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm

# --------- Настройки ----------
MODEL_NAME = "gpt2"  # или "gpt2-medium", "gpt2-xl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 5e-5

# --------- Данные ----------
# Примеры пар (input => target)
# Обычно ты бы читаешь их из JSONL!
examples = [
    {
        "source": "Fill in: [PERSON] founded [ORG] in [GPE].",
        "target": "Elon Musk founded SpaceX in the United States."
    },
    {
        "source": "Fill in: [PERSON] wrote [WORK].",
        "target": "J.K. Rowling wrote Harry Potter."
    }
]

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

tokenizer.pad_token = tokenizer.eos_token
model.eval()  # режим инференса

prompt = "Fill in: [PERSON] founded [ORG] in [GPE]."
model.to("cuda" if torch.cuda.is_available() else "cpu")


prompts = [
    "Fill in: [PERSON] founded [ORG] in [GPE].",
    "Fill in: [PERSON] wrote [WORK].",
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
#     num_return_sequences=1,
#     do_sample=True,  # для разнообразия
#     top_p=0.95,      # nucleus sampling
#     top_k=50         # или можно top_k=50
# )
    num_beams=3,       # beam search
    do_sample=False,   # beam делает лучше для фактов
    num_return_sequences=1
)

for i, out in enumerate(output):
    full_text = tokenizer.decode(out, skip_special_tokens=True)
    # Убираем prompt из результата, чтобы видеть только дополнение
    predicted = full_text[len(prompts[i]):].strip()
    print(f"\n🔹 PROMPT: {prompts[i]}\n🔹 OUTPUT: {predicted}")

# decoded = tokenizer.decode(output[0], skip_special_tokens=True)
# print(decoded)
