import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import classification_report

# Import Dataset
df = pd.read_csv('Data/clean_small.csv')

df = df[['review_body','target']]

class_names = ['negative','neutral','positive']

# Import Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Import Model (light-weight)
bert = BertModel.from_pretrained("bert-base-uncased")

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
    test_size=0.3,
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

BATCH_SIZE = 32 # Define Number of Batches

MAX_LEN = 64 # Define Max length for Sequences

# LOAD DATA FOR TRAIN, VALIDATION and TESTING.
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)

val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

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


num_classes = 3

model = Bert_Classifier(num_classes)

data = next(iter(train_data_loader))

input_ids = data['input_ids']

attention_mask = data['attention_mask'] 

output = model(input_ids,attention_mask)

# Define number of epochs
EPOCHS = 10

lr = 5e-5 # Set learning rate
optimizer = optim.AdamW(params=model.parameters(), lr=lr) # Pick Optimizer

# Calculate steps
total_steps = len(train_data_loader) * EPOCHS

# Initialize Scheduler
scheduler = get_linear_schedule_with_warmup(

    # Pass Optimizer
    optimizer, 

    # Specify number of warmup steps
    num_warmup_steps=0,

    # Pass number of steps
    num_training_steps=total_steps

)

loss_fn = nn.CrossEntropyLoss()

def train_epoch(model,data_loader,loss_fn,optimizer,scheduler,n_examples):

    model = model.train()

    losses = []

    correct_predictions = 0

    for d in data_loader:

        input_ids = d["input_ids"]

        attention_mask = d["attention_mask"]

        targets = d["targets"]

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)

        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, n_examples):

    model = model.eval()

    losses = []

    correct_predictions = 0

    with torch.no_grad():  

        for d in data_loader:

            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)

best_accuracy = 0
'''
for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')

    print('-' * 10)

    train_acc, train_loss = train_epoch(

        model,

        train_data_loader,

        loss_fn,

        optimizer,

        scheduler,

        len(df_train)

    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(

        model,

        val_data_loader,

        loss_fn,

        len(df_val)

    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')

    print()

    history['train_acc'].append(train_acc)

    history['train_loss'].append(train_loss)

    history['val_acc'].append(val_acc)

    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:

        torch.save(model.state_dict(), 'best_model_state.bin')

        best_accuracy = val_acc
'''

model = Bert_Classifier(3)

model.load_state_dict(torch.load('best_model_state.bin'))
'''
test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    len(df_test)
)

print(test_acc.item())

def get_predictions(model, data_loader):

    model = model.eval()

    review_texts = []

    predictions = []

    prediction_probs = []

    real_values = []

    with torch.no_grad():

        for d in data_loader:

            texts = d["review_text"]

            input_ids = d["input_ids"] 

            attention_mask = d["attention_mask"] 

            targets = d["targets"] 

            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask
            )


            _, preds = torch.max(outputs, dim=1)

            review_texts.extend(texts)

            predictions.extend(preds)

            prediction_probs.extend(outputs)

            real_values.extend(targets)

    predictions = torch.stack(predictions)

    prediction_probs = torch.stack(prediction_probs)

    real_values = torch.stack(real_values)

    return review_texts, predictions, prediction_probs, real_values

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(

  model,

  test_data_loader

)

print(classification_report(y_test, y_pred, target_names=['negative','neutral','positive']))
'''
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
