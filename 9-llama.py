import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


# --- TinyLlama setup ---
#llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
llm_model_name = "Felladrin/Llama-68M-Chat-v1"

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(llm_model_name)


sentiment_classifier = pipeline("sentiment-analysis")


def get_sentiment(text):
    result = sentiment_classifier(text)[0]
    label = result['label'].lower()
    if label == "negative":
        mood = "frustrated"
    elif label == "positive":
        mood = "happy"
    else:
        mood = "neutral"
    return mood


def build_prompt_old(user_history):
    prompt = f"""
You are a dialogue assistant. Based on the conversation history:
{user_history}

1) Summarize the user's latest message.
2) Detect user's mood.
3) Output dialogue state update and interface command in JSON format with fields:
- summary
- mood
- intent
- parameters

Example output:
{{
  "summary": "...",
  "mood": "...",
  "intent": "...",
  "parameters": {{}}
}}

Respond ONLY with the JSON.
User's latest message: 
"""

def build_prompt(system_prompt, conversation_history, user_message):
    prompt = f"<|system|>\n{system_prompt}</s>\n"
    for role, content in conversation_history:
        if role == "user":
            prompt += f"<|user|>\n{content}</s>\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}</s>\n"
    prompt += f"<|user|>\n{user_message}</s>\n<|assistant|>\n"
    return prompt


def build_prompt_llama_68m(system_message, conversation_history, user_message):
    """
    Create prompt for model: Felladrin/Llama-68M-Chat-v1
    <|im_start|>system
    ...
    <|im_end|>
    <|im_start|>user
    ...
    <|im_end|>
    <|im_start|>assistant
    """
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"

    for role, content in conversation_history:
        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt


def generate_llm_response_68m(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1].strip()
        text = text.split("<|im_end|>")[0].strip()
    return text


def handle_user_message_68m(system_prompt, conversation_history, user_message):
    # Build prompt
    prompt = build_prompt_llama_68m(system_prompt, conversation_history, user_message)

    # LLM response
    assistant_reply = generate_llm_response_68m(prompt)

    # Mood detection (по последнему user или ассистенту — можно выбрать)
    mood = get_sentiment(user_message)

    # Update history
    conversation_history.append(("user", user_message))
    conversation_history.append(("assistant", assistant_reply))
    return assistant_reply, mood, conversation_history


def generate_llm_response(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text


def handle_user_message(system_prompt, conversation_history, user_message):
    # Build prompt
    prompt = build_prompt(system_prompt, conversation_history, user_message)

    # LLM response
    assistant_reply = generate_llm_response(prompt)

    # Mood detection (по последнему user или ассистенту — можно выбрать)
    mood = get_sentiment(user_message)

    # Update history
    conversation_history.append(("user", user_message))
    conversation_history.append(("assistant", assistant_reply))
    return assistant_reply, mood, conversation_history


if __name__ == "__main__":

    system_prompt = ("You are helpful assistant to follow friendly dialog and answers user questions clearly.")

    conversation_history = []

    while True:
        user_message = input("user: ")
        # assistant_reply, mood, conversation_history = handle_user_message(
        #     system_prompt, conversation_history, user_message
        # )
        assistant_reply, mood, conversation_history = handle_user_message_68m(
            system_prompt, conversation_history, user_message
        )
        print(f"Assistant: {assistant_reply}")
        print(f"### Detected user-mood: {mood}")
