import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader,RandomSampler
from transformers import BertTokenizer, BertModel

# Import Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Import Model (light-weight)
bert = BertModel.from_pretrained("bert-base-uncased")

MAX_LEN = 512

class_names = ['Negative','Neutral','Positive']

class Bert_Classifier(nn.Module):

    # Define Constructor
    def __init__(self,classes):

        super(Bert_Classifier, self).__init__() # Initialize Parent Class

        self.bert = bert # Set Model 

        self.dropout = nn.Dropout(0.3) # Create dropout layer to prevent overfitting

        self.out = nn.Linear(self.bert.config.hidden_size, classes) # map hidden layer outputs to class probabilities

    # Define forward pass
    def forward(self, input_ids, attention_mask):
        
        _, pooled_output = self.bert(

            input_ids=input_ids,

            attention_mask=attention_mask,

            return_dict = False
        )

        output = self.dropout(pooled_output)

        return self.out(output)
    

model = Bert_Classifier(len(class_names))

model.load_state_dict(torch.load('best_model_state.bin'))

def classify(review_text):

    encoded_review = tokenizer.encode_plus(

        review_text,

        max_length=MAX_LEN,

        add_special_tokens=True,

        return_token_type_ids=False,

        padding='max_length',

        return_attention_mask=True,

        return_tensors='pt',

    )

    input_ids = encoded_review['input_ids']

    attention_mask = encoded_review['attention_mask']

    output = model(input_ids, attention_mask)

    _, prediction = torch.max(output, dim=1)

    return class_names[prediction]