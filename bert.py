import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Import Dataset
df = pd.read_csv('small_data.csv')

df = df[['review_body','target']]

# Import Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Import Model (light-weight)
model = BertModel.from_pretrained("bert-base-uncased")


# Pytorch Dataset
class tensorDataset(Dataset):

    # Initializes dataset
    def __init__(self, reviews, targets, tokenizer, max_len):

        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len


    # Returns length of dataset
    def __len__(self):

        return len(self.reviews)

    # Tokenizes and Encodes Input Returns Dictionary
    def __getitem__(self, item):

        review_text = str(self.reviews[item])
        target = self.targets[item]

        encoding = tokenizer.encode_plus(
  
            review_text, # Data for processing

            max_length=self.max_len, # Max length of sequences 

            add_special_tokens=True, # Includes CLS and SEP tokens for classification

            return_token_type_ids=False, # For Single Sequence Taks (Sentiment Analysis)

            padding='max_length', # Pads Sequences to max_length

            truncation=True,

            return_attention_mask=True, # Determines which tokens are meaningful

            return_tensors='pt',  # Return PyTorch tensors
        )
        return {

            'review_text': review_text,

            'input_ids': encoding['input_ids'].flatten(),

            'attention_mask': encoding['attention_mask'].flatten(),

            'targets': torch.tensor(target, dtype=torch.long)
        }

# Splitting Data into Train and Test samples
df_train, df_test = train_test_split(
  df, # Pass the df
  test_size=0.7, # 70-30 split
  random_state=2002, 
  stratify=df['target'] # balances classes
)

# Splitting test with validation set
df_val, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=2003,
)


def create_data_loader(df, tokenizer, max_len, batch_size):

    dataset = tensorDataset(
    
        reviews=df['review_body'].to_numpy(), # passes label to torch dataset

        targets=df['target'].to_numpy(), # passes target to torch dataset

        tokenizer=tokenizer, # passes specified tokenizer

        max_len=max_len # passes set max length 
    )

    return DataLoader(

        dataset, # Passing datatset

        batch_size=batch_size, # Defines batch size for data

        sampler = RandomSampler(dataset),

        # num_workers=4 Workers for Parallel Processing

    )

BATCH_SIZE = 16 # Define Number of Batches

MAX_LEN = 64 # Define Max length for Sequences

# LOAD DATA FOR TRAIN, VALIDATION and TESTING.
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)

val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))

print(data['input_ids'].shape)

print(data['attention_mask'].shape)

print(data['targets'].shape)

