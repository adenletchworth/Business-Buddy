import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch

# Import Tokenizer
bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Import BERT model
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Pytorch Dataset
class tensorDataset(Dataset):

    # Initializes dataset
    def __init__(self, text, emotion, tokenizer, max_len):

        self.text = text
        self.emotion = emotion
        self.tokenizer = tokenizer
        self.max_len = max_len


    # Returns length of dataset
    def __len__(self):
        return len(self.text)

    # Tokenizes and Encodes Input Returns Dictionary
    def __getitem__(self, item):

        text = str(self.reviews[item])
        emotion = self.targets[item]

        encoding = self.tokenizer.encode_plus(
  
            text, # Data for processing

            max_length=self.max_len, # Max length of sequences 

            add_special_tokens=True, # Includes CLS and SEP tokens for classification

            return_token_type_ids=False, # For Single Sequence Taks (Sentiment Analysis)

            padding='max_length', # Pads Sequences to max_length

            truncation=True,

            return_attention_mask=True, # Determines which tokens are meaningful

            return_tensors='pt',  # Return PyTorch tensors
        )
        return {

            'text': text,

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'targets': torch.tensor(emotion, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):

    dataset = tensorDataset(
    
        reviews=df['text'].to_numpy(), # passes label to torch dataset

        targets=df['emotion'].to_numpy(), # passes target to torch dataset

        tokenizer=tokenizer, # passes specified tokenizer

        max_len=max_len # passes set max length 
    )

    return DataLoader(

        dataset, # Passing datatset

        batch_size=batch_size, # Defines batch size for data

        sampler = RandomSampler(dataset),

        # num_workers=4 Workers for Parallel Processing

    )

BATCH_SIZE = 32 # Define Number of Batches

MAX_LEN = 64 # Define Max length for Sequences

# LOAD DATA FOR TRAIN, VALIDATION and TESTING.

df_train = pd.read_csv("./Data/emotions.csv")

df_test = pd.read_csv("test.txt",delimiter=";",header=None)

df_val = pd.read_csv("val.txt",delimiter=";",header=None)

train_data_loader = create_data_loader(df_train, bert_tokenizer, MAX_LEN, BATCH_SIZE)

val_data_loader = create_data_loader(df_val, bert_tokenizer, MAX_LEN, BATCH_SIZE)

test_data_loader = create_data_loader(df_test, bert_tokenizer, MAX_LEN, BATCH_SIZE)




