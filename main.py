import bert_classification as bc
from bert_topic import get_topics
import pandas as pd

df = pd.read_csv('Data/yelp_data.csv')

doc = df.text

topics = get_topics(doc)

print(topics)