import bert_classification as bc
from bert_topic import get_topics
import pandas as pd

df = pd.read_csv('Data/yelp_data.csv')

doc = df.text

topics = get_topics(doc)

topic_name = [name for name in topics['CustomName']]
doc_sentiment = [bc.classify(doc) for doc in topics['Representative_Docs']]

output = list(zip(topic_name,doc_sentiment))

print(doc for doc in topics['Representative_Docs'])


