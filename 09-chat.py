# 📌 simple_gpt2_chat.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Загружаем модель и токенизатор
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
model.eval()

# Важно: GPT-2 не имеет pad_token, задаём явно.
tokenizer.pad_token = tokenizer.eos_token  # Теперь паддинг и EOS одинаковы.

# 2. Инициализируем историю чата
history = ["System: You are a helpful assistant."]
history = []

print("🤖 Привет! Это GPT-2 чат-бот. Напиши что-то (exit для выхода).")

while True:
    user_input = input("Ты: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Бот: Пока 👋")
        break

    # Добавляем реплику пользователя
    history.append(f"User: {user_input}")
    # Формируем prompt
    prompt = "\n".join(history) + "\nBot:"

    # Токенизируем с attention_mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Генерация продолжения
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1
    )

    # Декодируем весь output → берём только новый кусок
    generated_text = tokenizer.decode(outputs[0])
    bot_reply = generated_text[len(prompt):].split("\n")[0].strip()

    # Добавляем ответ бота в историю
    history.append(f"Bot: {bot_reply}")

    print(f"Бот: {bot_reply}")
