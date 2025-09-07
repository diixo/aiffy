from nltk.stem import PorterStemmer
# or use WordNetLemmatizer


words = ["walk", "walked", "walking", "runs", "running", "ran"]

stemmer = PorterStemmer()
clusters = {}

for w in words:
    root = stemmer.stem(w)
    clusters.setdefault(root, []).append(w)

for root, group in clusters.items():
    print(f"{root}: {group}")


#################################################################

import morfessor

# Обучаем модель на словаре
model = morfessor.BaselineModel()
model.load_data("my_vocab.txt")  # файл с твоими словами
model.train_batch()

clusters = {}
words = ["walk", "walked", "walking", "runs", "running", "ran"]

for w in words:
    morphs = model.viterbi_segment(w)[0]  # список морфов
    root = morphs[0]  # берём первый морф как корень
    clusters.setdefault(root, []).append(w)

for root, group in clusters.items():
    print(f"{root}: {group}")
