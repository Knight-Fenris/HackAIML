# %%
import re

# %%
fname = 'text.txt'
fname_out = 'text_out.txt'

# %%
with open(fname, "r", encoding="utf-8") as f:
    text = f.read()
f.close()
print(type(text))

# %%
#remove foriegn language characters
res = re.sub(r'[^\x00-\x7f]',r'', text) 

#remove text having less than 2 words
text_2 = '\n'.join([i for i in res.split('\n') if len(i.split()) > 2])

with open(fname_out, "w", encoding="utf-8") as f:
    f.write(text_2)
f.close()

# print(text_2)

# str = "123456790 ABC#%? .(朱惠英)"
# result = re.sub(r'[^\x00-\x7f]',r'', str)
# print(result)

# %%
import spacy


# %%
nlp = spacy.load("en_core_web_sm")

# %%
nlp = spacy.load("en_core_web_sm")

# Example sentences
sentence1 = "cat"
sentence2 = "an animal"

# Process the sentences using spaCy
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)

# Compute the similarity between the two sentences
similarity = doc1.similarity(doc2)
print(f"Similarity between the two sentences: {similarity}")

# %%
s1 = nlp("Only 2 rooms left! 30 others are looking at this right now")
s2 = nlp("""False Urgency""")


# %%
s1.similarity(s2)


