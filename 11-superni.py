
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. Загружаем SuperNI
dataset = load_dataset("superni", split="train")  # можно указать split "validation" для теста

# Для примера возьмём первые 1000 элементов
small_dataset = dataset.select(range(1000))

# 2. Инициализируем токенизатор и модель GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# GPT-2 не имеет токена padding, установим pad_token равным eos_token
tokenizer.pad_token = tokenizer.eos_token

# 3. Подготовим данные — склеим instruction + input + output для обучения
def preprocess(example):
    # Формируем единый текст для токенизации: "Instruction: ... Input: ... Output: ..."
    text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
    tokenized = tokenizer(text, truncation=True, max_length=1024, padding="max_length")
    # GPT2LMHeadModel обучаем предсказывать следующий токен по всей последовательности
    # Поэтому labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = small_dataset.map(preprocess, batched=False)

# 4. Создаем DataCollator для MLM (на GPT-2 masking не нужен, но DataCollatorForLanguageModeling хорошо подходит)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Задаем параметры тренировки
training_args = TrainingArguments(
    output_dir="./gpt2-superni-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,  # если GPU поддерживает
    evaluation_strategy="no",
)

# 6. Создаем Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7. Запускаем обучение
trainer.train()

# 8. Сохраняем модель
trainer.save_model("./gpt2-superni-finetuned")
