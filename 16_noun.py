
import stanza

# Скачиваем английскую модель (только один раз)
stanza.download('en')

# Создаем NLP pipeline для английского
nlp = stanza.Pipeline('en')

sentence = "The boy hit the ball"

# Применяем NLP pipeline к предложению
doc = nlp(sentence)

# Вывод слов с их частями речи
for sent in doc.sentences:
    for word in sent.words:
        print(word.text, word.upos)

# Выделяем только существительные
nouns = [word.text for sent in doc.sentences for word in sent.words if word.upos == "NOUN"]
print("Nouns:", nouns)

###########################################################################################

# import spacy

# nlp = spacy.load("ru_core_news_sm")  # маленькая модель для русского
# doc = nlp("Мальчик ударил по мячу")

# nouns = [token.text for token in doc if token.pos_ == "NOUN"]
# print(nouns)  # ['Мальчик', 'мячу']

###########################################################################################

import nltk
from nltk import word_tokenize, pos_tag


nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


sentence = "The boy hit the ball"
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)

nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
print(nouns)  # ['boy', 'ball']

