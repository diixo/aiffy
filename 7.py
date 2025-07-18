
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Твой корпус: "Input: ... Output: ..."
# Создаёшь датасет
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="ner_train.txt",
    block_size=128
)

# Collator для LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# пример, как прямо сейчас сделать NER zero-shot через ChatGPT?

# В OpenAI GPT-3 обычно делают prompt-based NER zero-shot:
