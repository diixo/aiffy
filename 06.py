
import stanza


stanza.download('en')

nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

text = "Barack Obama was born in Hawaii and became the president of the United States."

# Analysis
doc = nlp(text)

for ent in doc.ents:
    print(f"Text: {ent.text}\tType: {ent.type}")

# Replace entities in text
output = text

# To avoid mixing of indices, go from end to begin
for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
    start = ent.start_char
    end = ent.end_char
    output = output[:start] + ent.type + output[end:]


print("\nReplaced тtext:")
print(output)
