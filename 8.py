
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import json


# with open('mask-fill-dataset.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line)
# dataset = Dataset.from_list(data)

dataset = load_dataset('json', data_files={'train': 'mask-fill-dataset.jsonl'})

def tokenize_function(examples):
    inputs = []
    labels = []

    for prompt, completion in zip(examples['prompt'], examples['completion']):
        # Конкатенируем
        full = prompt + " " + completion

        # Токенизируем
        enc = tokenizer(full, truncation=True, max_length=64)

        # Создаем labels: маскируем prompt
        input_len = len(tokenizer(prompt)['input_ids'])
        label_ids = [-100] * input_len + enc['input_ids'][input_len:]

        enc['labels'] = label_ids
        inputs.append(enc)

    # Преобразуем список словарей в один словарь
    return {
        'input_ids': [x['input_ids'] for x in inputs],
        'attention_mask': [x['attention_mask'] for x in inputs],
        'labels': [x['labels'] for x in inputs]
    }


# 1. Загружаешь токенизатор и модель
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. Добавляешь специальные токены
special_tokens_dict = {'additional_special_tokens': ['<PERSON>', '<ORG>', '<EVENT>', '<GPE>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

print(f"Добавлено {num_added_toks} токенов")

# 3. Resize эмбеддингов модели под новый словарь
model.resize_token_embeddings(len(tokenizer))



# 4. Токенизация
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=1024)

tokenized = dataset.map(tokenize)

# 5. Data collator для LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6. Trainer
training_args = TrainingArguments(
    output_dir="./fill_in_gpt2",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()
