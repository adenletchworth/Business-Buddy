from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
from umap import UMAP
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
df = pd.read_csv('Data/yelp_data.csv')

doc = df.text

# Import Models
embedded_model = SentenceTransformer("all-MiniLM-l6-v2")
hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
key_model = KeyBERTInspired()
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

embeddings = embedded_model.encode(doc)

topic_model = BERTopic(
    embedding_model=embedded_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=key_model,
)



topics, probs = topic_model.fit_transform(doc,embeddings)

print(topic_model.get_topic_info())


