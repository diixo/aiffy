
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import json


# 1. Загружаешь токенизатор и модель
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


# 2. Добавляешь специальные токены
special_tokens_dict = {'additional_special_tokens': ['<PERSON>', '<WORK>', '<ORG>', '<EVENT>', '<GPE>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
#print(f"Добавлено {num_added_toks} токенов")
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token


# with open('mask-fill-dataset.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line)
# dataset = Dataset.from_list(data)

# === 3. Загружаем датасет ===
dataset = load_dataset('json', data_files={'train': 'mask-fill-dataset.jsonl'})['train']


# === 4. Токенизация с маской ===
def tokenize_function(examples):
    # Батчево токенизируем объединённые строки prompt + completion
    inputs = tokenizer(
        [p + " " + c for p, c in zip(examples['prompt'], examples['completion'])],
        padding="max_length",      # добиваем до max_length
        truncation=True,           # режем длинные
        max_length=128,
        return_tensors=None        # вернём списки, не тензоры
    )

    labels = []
    for prompt, input_ids in zip(examples['prompt'], inputs['input_ids']):
        # Токенизируем prompt отдельно, чтобы узнать длину
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])

        # Создаём labels с -100 на позициях prompt
        label_ids = [-100] * prompt_len + input_ids[prompt_len:]

        # Добиваем до длины input_ids, если надо
        label_ids += [-100] * (len(input_ids) - len(label_ids))

        labels.append(label_ids)

    inputs['labels'] = labels
    return inputs


tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])

# === 5. Data collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# === 6. Тренировка ===
training_args = TrainingArguments(
    output_dir="./fill-in-gpt2",
    per_device_train_batch_size=8,
    num_train_epochs=50,
    learning_rate=3e-5,
    logging_steps=5,
    save_steps=500,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()


# Generation
prompt = "Fill-in: <PERSON> directed <WORK>"
encoded = tokenizer(prompt, return_tensors="pt", padding=True)
# print(tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist()))

# input_ids = tokenizer.encode(prompt, return_tensors="pt")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

output = model.generate(
    encoded['input_ids'],
    attention_mask=encoded["attention_mask"],
    max_new_tokens=20,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    #num_return_sequences=3,
    top_k=10,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

for o in output:
    print(tokenizer.decode(o, skip_special_tokens=True))