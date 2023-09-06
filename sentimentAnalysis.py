import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight

device = torch.device('cuda')

# Define pre-trained model
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Implement Bert Model for sentiment analysis
class BERT_Arch(nn.Module):
    def __init__(self,bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self,sent_id,mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
    
# Passing model to Bert implementation
model = BERT_Arch(bert)
model = model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(),lr =1e-5)

# Need Data Before Further implementation