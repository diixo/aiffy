
from keybert import KeyBERT

kw_model = KeyBERT("paraphrase-MiniLM-L6-v2")  # легкий BERT

# Quora Question Pairs (QQP) — вопросы, которые переформулируют один и тот же смысл.
# SNLI, MNLI — для понимания entailment и парафразирования.
# Сборки Paraphrase Mining из разных параллельных корпусов.

text = "New iPhone was announced in february"

# Extract ngram-range 2 = биграммы
keywords = kw_model.extract_keywords(
    text,
    keyphrase_ngram_range=(2, 2),
    stop_words=None
)

print(keywords)
