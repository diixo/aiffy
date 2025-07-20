
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

dataset = load_dataset('json', data_files={'train': 'mask-fill-dataset.jsonl'})['train']
print(len(dataset))

def tokenize_function(examples):

    input_texts = []
    labels = []

    # for instruction, response in zip(examples["prompt"], examples["completion"]):
    #     # Чёткий формат
    #     prompt_template = f"### Instruction:\n{instruction}\n\n### Response:\n\n"

    #     full_text = prompt_template + response

    #     # Токенизируем всё
    #     tokenized = tokenizer(full_text, truncation=True, max_length=256)

    #     # Длина части до Response:
    #     prompt_len = len(tokenizer(prompt_template, add_special_tokens=False)["input_ids"])

    #     input_ids = tokenized["input_ids"]

    #     # Маска для labels: prompt -> -100, completion -> реальные токены
    #     label_ids = [-100] * prompt_len + input_ids[prompt_len:]

    #     input_texts.append(input_ids)
    #     labels.append(label_ids)


    for instruction, response in zip(examples["prompt"], examples["completion"]):
        prompt_template = f"### Instruction:\n{instruction}\n\n### Response:\n"
        full_text = prompt_template + response

        # Токенизируем с паддингом и усечением
        tokenized = tokenizer(full_text, padding='max_length', truncation=True, max_length=256)

        prompt_len = len(tokenizer(prompt_template, add_special_tokens=False)["input_ids"])
        input_ids = tokenized["input_ids"]

        label_ids = [-100] * prompt_len + input_ids[prompt_len:]
        label_ids = label_ids[:len(input_ids)]

        input_texts.append(input_ids)
        labels.append(label_ids)


    return {"input_ids": input_texts, "labels": labels}


def format_input(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])

# === 5. Data collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# === 6. Тренировка ===
training_args = TrainingArguments(
    output_dir="./infill-gpt2",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    learning_rate=3e-5,
    logging_steps=5,
    save_steps=500,
    save_total_limit=1,
    weight_decay=0.001,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

trainer.train()


# Generation
prompt = format_input("Fill-in: <PERSON> directed <WORK>")
encoded = tokenizer(prompt, return_tensors="pt", padding=True)
# print(tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist()))

# input_ids = tokenizer.encode(prompt, return_tensors="pt")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def check_eos():
    print(tokenizer.eos_token)          # '<|endoftext|>'
    print(tokenizer.eos_token_id)       # 50256  (для GPT-2)
    print(tokenizer.decode([tokenizer.eos_token_id]))    # '<|endoftext|>'

do_sample = True

if do_sample:
    print(encoded['input_ids'])
    print(encoded["attention_mask"])

    output = model.generate(
        encoded['input_ids'],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=20,
        do_sample=do_sample,
        temperature=0.7,
        top_p=0.95,
        top_k=5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
else:
    output = model.generate(
        encoded['input_ids'],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=20,
        do_sample=do_sample,
        num_beams=4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

for o in output:
    print(tokenizer.decode(o, skip_special_tokens=True))