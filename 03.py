from flair.data import Sentence
from flair.models import SequenceTagger

# Загружаем готовую BERT-модель для NER
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")

sentence = Sentence("Barack Obama was born in Hawaii.")

# Предсказываем
tagger.predict(sentence)

print(sentence.to_tagged_string())
