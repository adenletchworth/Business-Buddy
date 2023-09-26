import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import nltk
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

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
    filtered_words = [word for word in lemmatized_words if word.lower() not in stop_words]

    processed_text = ' '.join(filtered_words)

    return processed_text

def add_stop_words(words):
    stop_words.extend(words)


reviews = list(map(process_text, df.review_body))







