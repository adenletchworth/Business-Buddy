from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

#nltk.download('stopwords') for my laptop
#nltk.download('punkt')

def processReviews(text):
    # Initialize stop words
    stop_words = set(stopwords.word('english')) 

    # Tokenize the words
    words = word_tokenize(text.lower())

    #Insert Lemmatizer or Stemmer