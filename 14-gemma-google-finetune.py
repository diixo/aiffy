
# https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/core/huggingface_text_full_finetune.ipynb

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from transformers import pipeline

from random import randint
import re


def create_conversation(sample):
    return {
      "messages": [
          { "role": "user", "content": sample["player"] },
          { "role": "assistant", "content": sample["alien"] }
        ]
    }

checkpoint_dir = "results"
base_model = "HuggingFaceTB/SmolLM-360M-Instruct"
base_model = "./models/gemma-3-270m-it"

learning_rate = 5e-5

npc_type = "martian" # ["martian", "venusian"]

# Load dataset from the Hub
dataset = load_dataset("datasets/MobileGame-NPC", npc_type, split="train")

# Convert dataset to conversational format
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

# Split dataset into 80% training samples and 20% test samples
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

# Print formatted user prompt
print(dataset["train"][0]["messages"])


### Fine-tune using TRL and the SFTTrainer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f"Device: {model.device}")
print(f"DType: {model.dtype}")



### Before fine-tune:
# The output below shows that the out-of-the-box capabilities may not be good enough for this use case.


# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"])-1)
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256)


# Extract the user query and original answer
print(f"Question:\n{test_sample['messages'][0]['content']}\n")
print(f"Original Answer:\n{test_sample['messages'][1]['content']}\n")
print(f"Generated Answer (base model):\n{outputs[0]['generated_text'][len(prompt):].strip()}")



outputs = pipe([{"role": "user", "content": "Sorry, you are a game NPC."}], max_new_tokens=256)
print(outputs[0]['generated_text'][1]['content'])


#########################
def test_message():
    message = [
        # give persona
        {"role": "system", "content": "You are a Martian NPC with a unique speaking style. Use an accent that replaces 's' sounds with 'z', uses 'da' for 'the', 'diz' for 'this', and includes occasional clicks like *k'tak*."},
    ]

    # few shot prompt
    for item in dataset['test']:
        message.append( {"role": "user", "content": item["messages"][0]["content"]} )
        message.append( {"role": "assistant", "content": item["messages"][1]["content"]} )

    # actual question
    message.append( {"role": "user", "content": "What is this place?"} )
    return message


outputs = pipe(test_message(), max_new_tokens=256)
print(outputs[0]['generated_text'])
print("-"*80)
print(outputs[0]['generated_text'][-1]['content'])

############################################################################################

# Training

from trl import SFTConfig

torch_dtype = model.dtype

args = SFTConfig(
    output_dir=checkpoint_dir,              # directory to save and repository id
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=5,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every step
    save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # Template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)

from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)

trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

########################################################################

import matplotlib.pyplot as plt

# Access the log history
log_history = trainer.state.log_history

# Extract training / validation loss
train_losses = [log["loss"] for log in log_history if "loss" in log]
epoch_train = [log["epoch"] for log in log_history if "loss" in log]
eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

# Plot the training loss
plt.plot(epoch_train, train_losses, label="Training Loss")
plt.plot(epoch_eval, eval_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()

#############################################################

# ** validation loss >> training loss: overfitting
# ** validation loss > training loss: some overfitting
# ** validation loss < training loss: some underfitting
# ** validation loss << training loss: underfitting

# Test Model Inference #####################################

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = checkpoint_dir

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def test(test_sample):
    # Convert as test example into a prompt with the Gemma template
    prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, disable_compile=True)

    # Extract the user query and original answer
    print(f"Question:\n{test_sample['messages'][0]['content']}")
    print(f"Original Answer:\n{test_sample['messages'][1]['content']}")
    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
    print("-"*80)

# Test with an unseen dataset
for item in dataset['test']:
    test(item)
