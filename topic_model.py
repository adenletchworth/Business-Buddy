import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim.models.phrases import Phrases, Phraser
import re
import pyLDAvis.gensim_models

filename = './Data/yelp_data.csv'

df = pd.read_csv(filename)

# Initialize Lemmatizer
lz = WordNetLemmatizer()

# Initialize Stop words
stop_words = stopwords.words("english")

# Takes Input, removes punctuation and tokenizes
def process_text(text):
    
    # Remove punctuation from text
    text = re.sub(r'[_"\-;%()|+&=*%.,!?\'\":#$@\[\]/]', ' ', text)
    text = re.sub(r'\b\w{1,2}\b',' ',text)

    # Tokenizes input text
    words = word_tokenize(text)

    return words

# Get LoL of tokens
tokens = list(map(process_text, df.text))

# Build the bigram and trigram models
bigram = Phrases(tokens, min_count=5, threshold=100) # higher threshold fewer phrases.
#trigram = Phrases(bigram[tokens], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
#trigram_mod = Phraser(trigram)

def process_tokens(tokens):

    # lowercases and filters words
    filtered_words = [word.lower() for word in tokens if word.lower() not in stop_words]

    bigram_doc = bigram_mod[filtered_words]

    # Lemmatizes input text
    processed_text = [lz.lemmatize(word) for word in bigram_doc]

    return processed_text

# Add stop words to be filtered out
def add_stop_words(words):

    stop_words.extend(words)

# creates a list of lists of stop words
reviews = list(map(process_tokens, tokens))

# Create Dictionary
id2word = Dictionary(reviews)

# Create Corpus
corpus = [id2word.doc2bow(text) for text in reviews]

# Initialize LDA
lda = LdaModel(corpus,10,id2word)

# Initialize Coherence Model
coherence_model_lda = CoherenceModel(model=lda, texts=reviews, dictionary=id2word,coherence='u_mass')

# Get coherence score
coherence_lda = coherence_model_lda.get_coherence()

print('Coherence Score: ', coherence_lda)


# Add labels to the pyLDAvis visualization
vis = pyLDAvis.gensim_models.prepare(lda, 
                                     corpus, 
                                     dictionary=lda.id2word)  



pyLDAvis.save_html(vis, 'lda_visualization.html')



