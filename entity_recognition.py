import spacy
from spacy import displacy
import pandas as pd

nlp = spacy.load('en_core_web_trf')

test = 'George Bush went to the Met Gala and decided that mcdonalds was the place to be'

doc = nlp(test)

for entity in doc.ents:
    print(entity.text,entity.label_)

displacy.serve(doc, style='ent', page=True, minify=True)