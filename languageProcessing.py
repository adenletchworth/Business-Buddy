from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
 
# Give function text, outputs processed text as array of words
def processReviews(text):
    
    
    # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english')) 
    lz = WordNetLemmatizer()

    # Tokenize the words
    words = word_tokenize(text)

    # get words ready for processing, lemmatizer, lowercase, and remove stop words
    processed_review = [lz.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]

    # returns array of processed words
    return processed_review

# Implement Bert Model for sentiment analysis
class BERT_Arch(nn.Module):
    




text = 'This product was so good, although the bearrings were loose'
proccesed_text = processReviews(text)
print(proccesed_text)