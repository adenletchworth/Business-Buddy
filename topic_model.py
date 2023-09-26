import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import nltk
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

df = pd.read_csv('Data/small_data.csv')

# Initialize Lemmatizer
lz = WordNetLemmatizer()

# Initialize Tokenizer
tokenizer = nltk.tokenize.WordPunctTokenizer()

# Initialize Stop words
stop_words = stopwords.words("english")


# Takes Input text and outputs processed input text
def process_text(text):

    # Tokenizes input text
    words = tokenizer.tokenize(text)

    # Lemmatizes input text
    lemmatized_words = [lz.lemmatize(word) for word in words]

    # lowercases and filters words
    filtered_words = [word.lower() for word in lemmatized_words if word.lower() not in stop_words]

    processed_text = ' '.join(filtered_words)

    return processed_text

# Add stop words to be filtered out
def add_stop_words(words):

    stop_words.extend(words)

# creates a list of lists of stop words
reviews = [list(map(process_text, df.review_body))]

# Create Dictionary
id2word = Dictionary(reviews)

# Create Corpus
corpus = [id2word.doc2bow(text) for text in reviews]

# Initialize LDA
lda = LdaModel(corpus,5,id2word)

# Initialize Coherence Model
coherence_model_lda = CoherenceModel(model=lda, texts=reviews, dictionary=id2word,coherence='u_mass')

# Get coherence score
coherence_lda = coherence_model_lda.get_coherence()

print('Coherence Score: ', coherence_lda)

# NEED TO IMPLEMENT OPTIMAL NUMBER OF LABELS, NEED FINAL DATA