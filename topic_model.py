import pandas as pd
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
import re
import pyLDAvis.gensim_models
import spacy

filename = './Data/small_data.csv'

df = pd.read_csv(filename)


# Initialize Spacy Processor
nlp = spacy.load('en_core_web_sm',disable=['parser','ner'])

# Initialize Stop words
stop_words = stopwords.words("english")

# Define Hyperparameters
ALPHA = 0.1
BETA = 0.1
NUM_TOPICS = 5

# Add stop words to be filtered out
add_stop_words = lambda words: stop_words.extend(words)

# Lemmatizes text and filters according to POS
def get_lemma(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    processed_text = []
    for text in texts:
        doc = nlp(text)
        processed_text.append(" ".join(token.lemma_ for token in doc if token.pos_ in allowed_postags))
    return processed_text

# Processes text using spacy preprocess
def filter_tokens(texts):
    processed_text = []
    for text in texts:
        processed = simple_preprocess(text,deacc=True)
        processed_text.append(processed)
    return processed_text

# Pass to lemma function
lemma = get_lemma(df.review_body)

# Pass to spacy process function
final_data=  filter_tokens(lemma)
        
'''
# Build the bigram and trigram models
bigram = Phrases(tokens, min_count=5, threshold=10e-5) # higher threshold fewer phrases.
trigram = Phrases(bigram[tokens], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)
'''

# Create Dictionary
id2word = Dictionary(final_data)

# Create Corpus
corpus = [id2word.doc2bow(text) for text in final_data]

# Initialize LDA
lda = LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=NUM_TOPICS,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha='auto'
)

# Initialize Coherence Model
coherence_model_lda = CoherenceModel(
    model=lda, 
    texts=final_data, 
    dictionary=id2word,
    coherence='u_mass'
)

# Get coherence score
coherence_lda = coherence_model_lda.get_coherence()

print('Coherence Score: ', coherence_lda)


# Add labels to the pyLDAvis visualization
vis = pyLDAvis.gensim_models.prepare(lda, 
                                     corpus, 
                                     dictionary=lda.id2word)  



pyLDAvis.save_html(vis, 'lda_visualization.html')



