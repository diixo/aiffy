
# github.com/allenai/natural-instructions
# huggingface.co/datasets/super_natural_instructions

from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling


# === 1. Train split ===
train_dataset = load_dataset("allenai/natural-instructions", split="train")

# === 2. Validation split ===
# В SuperNI есть "validation", можно использовать его:
val_dataset = load_dataset("allenai/natural-instructions", split="validation")

# === 3. Токенизатор и модель ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# === 4. Препроцессинг ===
def preprocess(example):
    text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
    enc = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

train_tokenized = train_dataset.map(preprocess, batched=False)
val_tokenized = val_dataset.map(preprocess, batched=False)

# === 5. Data collator ===
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === 6. TrainingArguments ===
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=200,
    save_steps=2000,
    save_total_limit=3,
    evaluation_strategy="steps",  # <-- Важно! Запускать валидацию каждые eval_steps
    eval_steps=2000,              # <-- Частота валидации
    fp16=True,
    #gradient_accumulation_steps=2,
    #dataloader_num_workers=2,
    report_to="none",
    lr_scheduler_type="constant",
    save_strategy="no",
)

# === 7. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,     # <-- подключаем validation
    tokenizer=tokenizer,
    data_collator=collator,
)

# === 8. Train ===
trainer.train()


# === 9. Сохраняем модель ===
trainer.save_model("./gpt2-ft-superni")
