import pandas as pd
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, TfidfModel
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
NUM_TOPICS = 10

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
text_data=  filter_tokens(lemma)
        


bigram = Phrases(
    text_data, 
    min_count=5, 
    threshold=50
) 

trigram = Phrases(
    bigram[text_data], 
    threshold=50
)  

bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

def bigram_to_text(texts):
    return [bigram[doc] for doc in texts]

def trigram_to_text(texts):
    return [trigram[bigram[doc]] for doc in texts]
    
data_bigrams = bigram_to_text(text_data)

data_trigrams = trigram_to_text(data_bigrams)

id2word= Dictionary(data_trigrams)

corpus = [id2word.doc2bow(text) for text in data_trigrams]

tfidf = TfidfModel(
    corpus=corpus,
    id2word=id2word
)

# Set threshold for 'valuable' words
LOW = 0.03

words = []

missing_words = []

for i in range(0,len(corpus)):
    bow=corpus[i]
    low_words = []

    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]

    low_words = [id for id, value in tfidf[bow] if value < LOW]
    drops = low_words+missing_words

    for item in drops:
        words.append(id2word[item])

    missing_words = [id for id in bow_ids if id not in tfidf_ids]

    new_bow = [b for b in bow if b[0] not in low_words and b[0] not in missing_words]
    corpus[i]=  new_bow
'''
# Create Dictionary
id2word = Dictionary(text_data)

# Create Corpus
corpus = [id2word.doc2bow(text) for text in text_data]
'''

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
    texts=text_data, 
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



