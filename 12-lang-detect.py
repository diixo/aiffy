
from langdetect import detect
lang = detect("Je suis fatigué.") 

print(lang)


#################################

from transformers import pipeline

classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
results = classifier("Ciao, come stai?")

print(results)
