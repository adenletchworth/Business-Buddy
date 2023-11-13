import bert_classification as bc
from bert_topic import get_topics
import pandas as pd

df = pd.read_csv('Data/small_data.csv')

doc = df.review_body

# Assuming get_topics returns a dictionary
topics_data = get_topics(doc)

# Assuming 'CustomName' and 'Representative_Docs' are keys in the dictionary
print(topics_data.columns)
topics = pd.DataFrame(topics_data)

# Assuming bc is a sentiment classifier
topics['doc_sentiment'] = [bc.classify(doc) for doc in topics_data['Representative_Docs']]

# Save to CSV
topics.to_csv('output_topics.csv', index=False)
