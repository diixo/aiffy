# aiffy

| Description (task)                                       | Model                                                                                                              | File                                 |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| Zero-shot-classification: select tags and order by score | facebook/bart-large-mnli                                                                                           | [01.py](01.py)                       |
| Extract ngrams (bigramm) with ordering by priorities     | paraphrase-MiniLM-L6-v2                                                                                            | [02.py](02.py)                       |
| Extract NERs from text                                   | flair/ner-english-ontonotes-fast                                                                                   | [03.py](03.py)                       |
| Extract NERs from text (with scores) - full extraction   | dslim/bert-base-NER                                                                                                | [04.py](04.py)                       |
| Extract NERs from text (with scores) - full extraction   | flair/ner-english                                                                                                  | [05.py](05.py)                       |
| Ner extraction                                           | stanza.Pipeline(lang='en', processors='tokenize,ner')                                                              | [06.py](06.py)
| Nexr extraction with gpt2 (from text file) - seq2seg     | pretrained gpt2                                                                                                    | [07.py](07.py)|
