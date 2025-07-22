
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# --- Настройка LLM для диалоговой генерации (пример с GPT-подобной моделью) ---
llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # замените на свою
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(llm_model_name)

# Генератор текста из LLM (простейший)
def generate_llm_response(prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# --- Sentiment analysis (можно заменить на любую другую, более точную) ---
sentiment_classifier = pipeline("sentiment-analysis")

def get_sentiment(text):
    result = sentiment_classifier(text)[0]
    # Приведём к общему формату
    label = result['label'].lower()
    if label == "negative":
        mood = "frustrated"
    elif label == "positive":
        mood = "happy"
    else:
        mood = "neutral"
    return mood

# --- Пример prompt для LLM с инструкцией выдать JSON DST + summary ---
def build_prompt(user_history):
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
    return prompt

# --- Обработка нового сообщения от пользователя ---
def handle_user_message(user_history, latest_message):
    # 1. Обновляем историю
    updated_history = user_history + "\nUser: " + latest_message + "\nAssistant:"

    # 2. Формируем prompt для LLM
    prompt = build_prompt(updated_history)

    # 3. Получаем JSON-ответ от LLM
    llm_output = generate_llm_response(prompt)

    # Попытка извлечь JSON из текста LLM (очень простой способ)
    try:
        json_start = llm_output.index("{")
        json_str = llm_output[json_start:]
        llm_json = json.loads(json_str)
    except Exception as e:
        # На случай ошибки парсинга
        llm_json = {
            "summary": "Failed to parse LLM output.",
            "mood": get_sentiment(latest_message),
            "intent": "unknown",
            "parameters": {}
        }

    # 4. На всякий случай можно обновить настроение, если LLM "зашло" не точно
    if llm_json.get("mood", "") == "unknown":
        llm_json["mood"] = get_sentiment(latest_message)

    # 5. Возвращаем результат и обновлённую историю
    return llm_json, updated_history

# --- Пример использования ---
if __name__ == "__main__":
    user_history = ""
    while True:
        latest_message = input("User: ")
        response_json, user_history = handle_user_message(user_history, latest_message)
        print("Assistant JSON response:")
        print(json.dumps(response_json, indent=2))
