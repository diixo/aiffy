import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import re


Train = True
model_path = "./infill-gpt2"


def check_special_tokens():
    print(f"eos_token: {tokenizer.eos_token}={tokenizer.eos_token_id}, decode={tokenizer.decode([tokenizer.eos_token_id])}")
    print(f"pad_token_id: {tokenizer.pad_token}={tokenizer.pad_token_id}, decode={tokenizer.decode([tokenizer.pad_token_id])}")


# 1. Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


if os.path.isdir(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    Train = False
else:
    model = GPT2LMHeadModel.from_pretrained("gpt2")


# 2. Add special tokens
#special_tokens_dict = {'additional_special_tokens': ['<PERSON>', '<WORK>', '<ORG>', '<EVENT>', '<GPE>'], 'pad_token': '<PAD>'}
special_tokens_dict = {'additional_special_tokens': ['<PERSON>', '<WORK>', '<ORG>', '<EVENT>', '<GPE>']}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


# with open('mask-fill-dataset.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line)
# dataset = Dataset.from_list(data)

dataset = load_dataset('json', data_files={'train': 'mask-fill-dataset.jsonl'})['train']
#print(len(dataset))


"""
def tokenize_function(examples):
    inputs = []
    labels = []

    for prompt, completion in zip(examples['prompt'], examples['completion']):
        full = prompt + " " + completion

        enc = tokenizer(full, truncation=True, max_length=64)

        input_len = len(tokenizer(prompt)['input_ids'])
        label_ids = [-100] * input_len + enc['input_ids'][input_len:]

        enc['labels'] = label_ids
        inputs.append(enc)

    return {
        'input_ids': [x['input_ids'] for x in inputs],
        'attention_mask': [x['attention_mask'] for x in inputs],
        'labels': [x['labels'] for x in inputs]
    }
"""

def tokenize_function(examples, max_length=128):

    input_texts = []
    labels = []

    # for instruction, response in zip(examples["prompt"], examples["completion"]):
    #     # Чёткий формат
    #     prompt_template = f"### Instruction:\n{instruction}\n\n### Response:\n\n"

    #     full_text = prompt_template + response

    #     # Токенизируем всё
    #     tokenized = tokenizer(full_text, truncation=True, max_length=max_length)

    #     # Длина части до Response:
    #     prompt_len = len(tokenizer(prompt_template, add_special_tokens=False)["input_ids"])

    #     input_ids = tokenized["input_ids"]

    #     # Маска для labels: prompt -> -100, completion -> реальные токены
    #     label_ids = [-100] * prompt_len + input_ids[prompt_len:]

    #     input_texts.append(input_ids)
    #     labels.append(label_ids)


    for instruction, response in zip(examples["prompt"], examples["completion"]):
        prompt_template = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # automatic padding is not real EOS-token inside target. We force add eos_token to the end
        full_text = prompt_template + response + tokenizer.eos_token

        input_ids = tokenizer(full_text, padding='max_length', truncation=True, max_length=max_length)["input_ids"]

        prompt_len = len(tokenizer(prompt_template, add_special_tokens=False)["input_ids"])

        len_real = len(tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"])

        # only completion take part in loss
        label_ids = [-100] * prompt_len + input_ids[prompt_len:len_real]

        # Расширяем label_ids значениями -100, если они короче input_ids
        if len(label_ids) < len(input_ids):
            label_ids += [-100] * (len(input_ids) - len(label_ids))
        else:
            label_ids = label_ids[:len(input_ids)]

        input_texts.append(input_ids)
        labels.append(label_ids)

    return {"input_ids": input_texts, "labels": labels}


def format_input(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['prompt', 'completion'])

# === 5. Data collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# === 6. training ===
if Train:
    training_args = TrainingArguments(
        output_dir=model_path,
        per_device_train_batch_size=4,
        num_train_epochs=50,
        learning_rate=1e-5,
        logging_steps=5,
        save_strategy="no",
        save_total_limit=1,
        lr_scheduler_type="constant",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    trainer.model.save_pretrained(model_path)


# Generation
prompt = format_input("Fill-in: <PERSON> directed <WORK>")
encoded = tokenizer(prompt, return_tensors="pt", padding=False)
# print(tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist()))

# input_ids = tokenizer.encode(prompt, return_tensors="pt")
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


do_sample = True

if do_sample:
    output = model.generate(
        encoded['input_ids'],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=10,
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
        max_new_tokens=10,
        do_sample=do_sample,
        num_beams=4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


for ids in output:
    txt = tokenizer.decode(ids, skip_special_tokens=False)
    txt = re.sub(r' {2,}', ' ', txt)
    print(txt)
    #print([tokenizer.decode([tid]) for tid in ids])
