from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic.representation import KeyBERTInspired,MaximalMarginalRelevance,PartOfSpeech
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

### Method for Getting Topic Models from Text ###

def get_topics(doc):
        
        ## Initialize Models ##

        # For Encoding Doccuments / Pre Processing
        embedded_model = SentenceTransformer("all-MiniLM-l6-v2")

        # For Controlling Number of Topics
        hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        # For Removing Stop Words, Post Encoding
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

        # For Reducing Size of Embeddings
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

        # For Different Representations 
        key_model = KeyBERTInspired()
        pos_model = PartOfSpeech("en_core_web_sm")
        mmr_model = MaximalMarginalRelevance(diversity=0.3)

        # Make Dictionary of Different Representations
        representation_model={
                "KeyBERT":key_model,
                "POS":pos_model,
                "MMR":mmr_model
        }


        # Encode Doccuments with SentenceTransformer
        embeddings = embedded_model.encode(doc)

        # Initialize our Topic Model
        topic_model = BERTopic(
                embedding_model=embedded_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model
        )

        # Fit our doc and encoded doc to topic model
        topics, probs = topic_model.fit_transform(doc,embeddings)

        # Automatically generate labels using KeyBERT
        keybert_topic_labels = {topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()}
        topic_model.set_topic_labels(keybert_topic_labels)

        return topic_model.get_topic_info()




