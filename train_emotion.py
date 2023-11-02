from collections import defaultdict
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
import pandas as pd
import torch


# Import Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Import BERT model
model = DistilBertModel.from_pretrained("distilbert-base-uncased",output_hidden_states=True)

# LOAD DATA FOR TRAIN, VALIDATION and TESTING.
df_train = pd.read_csv("./Data/emotions.csv")
df_test = pd.read_csv("./Data/emotions_test.csv")
df_val = pd.read_csv("./Data/emotions_val.csv")

class EmotionDataset(torch.utils.data.Dataset):

    def __init__(self, emotion_data):
        self.labels = emotion_data.emotion
        self.text = emotion_data.text

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self,index):

        text_item = self.text[index]
        label_item = self.labels[index]

        encoding= tokenizer.encode_plus(
            text_item,
            padding='max_length',
            max_length=32,
            truncation=True,
            return_tensors='pt'
        )

        return {
           'text': text_item,
           'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label_item,dtype=torch.int)
        }

def create_data_loader(df, batch_size):

    dataset = EmotionDataset(df)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        sampler=torch.utils.data.RandomSampler(dataset),
    )

class BertClassifier(torch.nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = model  
        self.dropout = torch.nn.Dropout(p=dropout) 
        self.linear = torch.nn.Linear(768, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output[0])
        linear_output = self.linear(dropout_output)
        output_layer = self.relu(linear_output)
        return output_layer


