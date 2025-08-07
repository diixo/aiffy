
from flair.data import Sentence
from flair.models import SequenceTagger

# Загружаем готовый теггер NER
tagger = SequenceTagger.load("flair/ner-english")

# Текст
sentence = Sentence("George Washington went to Washington.")

# Предсказание
tagger.predict(sentence)

# Выводим разметку
print(sentence.to_tagged_string())
