
# Multi-Genre Natural Language Inference


from transformers import pipeline

# Загружаем Zero-Shot классификатор
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#classifier = pipeline("zero-shot-classification", model="textattack/roberta-base-MNLI")


text = "New iPhone was announced in february"

candidate_labels = ["Device", "Technology", "Sport", "Politics"]

result = classifier(text, candidate_labels, multi_label=True)

print(result)

