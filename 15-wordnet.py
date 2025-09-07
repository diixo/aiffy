
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')  # для многоязычных определений


from nltk.corpus import wordnet as wn


synsets = wn.synsets('ability')
print(f"Found {len(synsets)} senses for 'dragon':\n")

for i, syn in enumerate(synsets, 1):
    print(f"{i}. Name: {syn.name()}")
    print(f"   Definition: {syn.definition()}")
    print(f"   Examples: {syn.examples()}")
    print(f"   Lemmas: {[lemma.name() for lemma in syn.lemmas()]}")
    print()


###################################################################

# Semantic net: “dog → IsA → animal”, “knife → UsedFor → cutting”

import requests
import time

# Например, хотим получить связи для слова "sword"
concept = "ball"
url = f"http://api.conceptnet.io/c/en/{concept}"

response = requests.get(url).json()

# Смотрим первые 5 ребер (edges)
for edge in response['edges'][:10]:
    start = edge['start']['label']
    end = edge['end']['label']
    relation = edge['rel']['label']
    print(f"{start} --{relation}--> {end}")
    time.sleep(3)


dataset = []
for edge in response['edges'][:10]:
    relation = edge['rel']['label']
    if relation in ["/rUsedFor", "CapableOf"]:
        dataset.append({
            "item": concept,
            "relation": relation,
            "action": edge['end']['label']
        })
    time.sleep(3)

###################################################################

import stanza

# Загрузка модели для русского языка
stanza.download('ru')

nlp = stanza.Pipeline('ru')
doc = nlp("Мальчик ударил по мячу")

# Проход по словам и их POS
for sent in doc.sentences:
    for word in sent.words:
        print(word.text, word.upos)
