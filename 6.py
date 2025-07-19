
import stanza

# Скачиваем модель
stanza.download('en')

# Создаем пайплайн с NER
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

# Исходный текст
text = "Barack Obama was born in Hawaii and became the president of the United States."

# Анализируем
doc = nlp(text)

# Посмотрим, что нашлось
for ent in doc.ents:
    print(f"Text: {ent.text}\tType: {ent.type}")

# Заменим сущности в тексте
output = text

# Чтобы не напутать индексы, идем от конца текста
for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
    start = ent.start_char
    end = ent.end_char
    output = output[:start] + ent.type + output[end:]

print("\nЗаменённый текст:")
print(output)
