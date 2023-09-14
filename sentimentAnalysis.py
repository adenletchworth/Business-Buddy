import numpy as np
from datetime import datetime,time
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.optim import Adam
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler

#device = torch.device('cuda')

# Import CSV for train/validation
df = pd.read_csv('pamazonProducts.csv')

# Training Set
train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'],df['label'],
                                                                    random_state = 2002,
                                                                    test_size = 0.3,
                                                                    stratify=df['label'])
# Validation Set
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)

# Define pre-trained model
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# Tokenize and Encode Training Set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

# Tokenize and Encode Validation Set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)

# Tokenize and Encode Testing Set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True
)


train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)

val_sampler = SequentialSampler(val_data)

val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

for param in bert.parameters():
    param.requires_grad = False

# Implement Bert Model for sentiment analysis
class BERT_Arch(nn.Module):
    def __init__(self,bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,2)
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
#model = model.to(device)

# Define optimizer
optimizer = Adam(model.parameters(),lr=1e-5)

# Need Data Before Further implementation
y = np.stack((train_labels))
class_weights = compute_class_weight('balanced',classes=np.unique(y),y=y)
print("Class Weights:",class_weights)

weights = torch.tensor(class_weights,dtype=torch.float)
cross_entropy = nn.NLLLoss(weight=weights)

epochs = 10


