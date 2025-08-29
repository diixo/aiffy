
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset


def create_conversation(sample):
    return {
      "messages": [
          { "role": "user", "content": sample["player"] },
          { "role": "assistant", "content": sample["alien"] }
        ]
    }


base_model = "models/gemma-3-270m-it"
# @param [
# "google/gemma-3-270m-it",
# "google/gemma-3-1b-it",
# "google/gemma-3-4b-it",
# "google/gemma-3-12b-it",
# "google/gemma-3-27b-it"
# ] {"allow-input":true}

#checkpoint_dir = "/content/drive/MyDrive/MyGemmaNPC" #@param {type:"string"}

learning_rate = 5e-5

npc_type = "martian" # ["martian", "venusian"]

# Load dataset from the Hub
dataset = load_dataset("bebechien/MobileGameNPC", npc_type, split="train")

# Convert dataset to conversational format
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

# Split dataset into 80% training samples and 20% test samples
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

# Print formatted user prompt
print(dataset["train"][0]["messages"])


### Fine-tune Gemma using TRL and the SFTTrainer


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

