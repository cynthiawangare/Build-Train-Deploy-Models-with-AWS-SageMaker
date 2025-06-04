import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd


s3_path = 's3://hugging-face-multiclasstextclassification1/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URI', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[['TITLE', 'CATEGORY']]

my_dict = {
    'e': 'Entertainment',
    'b': 'Business',
    't': 'Science',
    'm': 'Health'
}  # We defined the above categories in the Exploratory Data Analysis Notebook


def update_cat(x):
    return my_dict[x]


df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))

# Train the model with a small subset of data to test if the training jobs was successful.

#  Sometimes training jobs can fail and this is why you test. That way you save on cost if the training job fails.

df = df.sample(frac=0.05, random_state=1)  # frac: fraction of data we use: 0.05

df = df.reset_index(drop=True)


encode_dict = {}


def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]


df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))  # creates a new column from the category column

df = df.reset_index(drop=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.iloc[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index, 2], dtype=torch.long)
        }

    def __len__(self):
        return self.len


train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)  # use 80% of our dataset to train
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)  # drop 20% of remaining dataset to make the test dataset

train_dataset.reset_index(drop=True)

print("Full dataset:{}".format(df.shape))
print("Train dataset:{}".format(train_dataset.shape))   # print train dataset
print("Test dataset:{}".format(test_dataset.shape))   # print test dataset

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

training_set = NewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, tokenizer, MAX_LEN)

train_parameters = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

test_parameters = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

training_loader = DataLoader(training_set, **train_parameters)
testing_loader = DataLoader(testing_set, **test_parameters)


class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')  # loads a pre-trained distilbertmodel as base model with uncased vocubualry. we only need to fine-tune 

        self.pre_classifier = torch.nn.Linear(768, 768)  # add more weights to model to allow more learning 

        self.dropout = torch.nn.Dropout(0.3)  # randomly drop neurons to prevent overfitting

        self.classifier = torch.nn.Linear(768, 4)  # take 768 inputs and output the 4 categories of news we had, my_dict

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]  # represents fnal layer of the model. output of data after going thru the distilbert model
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)  # ReLU Activation function. helps with the vanishing gradient problem
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def calculate_accu(big_idx, targets):  # here we test accuracy of model outputs to the categories given
    n_correct = (big_idx == targets).sum().item()  # it will show accuracy with true and false statements. the sum will give total number of truths and fasle outputs

    return n_correct


def train(epoch, model, device, training_loader, optimizer, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)  # know which token to pay attention to or its just padding
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask)

        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accu(big_idx, targets)  # no of correct predictions
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            # to get the accuracy of the model we will do n_correct(no. of correct predictions) divide by nb_tr_examples(no. of true examples)
            print(f"Training loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
# we are the using the atom optimizer for gadient descent and back propagation
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training accuracy Epoch: {epoch_accu}")

    return


def valid(epoch, model, testing_loader, device, loss_function):  # function for the validation set, typically smaller than training set
    model.eval()

    n_correct = 0
    tr_loss = 0
    nb_tr_examples = 0
    nb_tr_steps = 0

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accu(big_idx, targets)
            nb_tr_steps +=1
            nb_tr_examples += targets.size(0)
            if _ % 1000 == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation loss per 1000 steps: {loss_step}")
                print(f"Validation accuracy per 1000 steps: {accu_step}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation loss per Epoch: {epoch_loss} at epoch {epoch}")
    print(f"Validation accuracy epoch: {epoch_accu} at epoch {epoch}")

    return


def main():
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)  # training-related parameters 
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    args = parser.parse_args()

    args.epochs
    args.train_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set either gpu or cpu for training
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBERTClass()
    model.to(device)

    LEARNING_RATE = 1e-05
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    loss_function = torch.nn.CrossEntropyLoss()

    # train loop. start of training. for every epoch

    EPOCHS = 2

    for epoch in range(EPOCHS):
        print(f"starting epoch: {epoch}")

        train(epoch, model, device, training_loader, optimizer, loss_function)

        valid(epoch, model, testing_loader, device, loss_function)

    output_dir = os.environ['SM_MODEL_DIR']  # sagemakers designated output directory. output to s3 bucket
    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_news.bin')
    output_vocab_file = os.path.join(output_dir, 'vocab_distilbert_news.bin')

    torch.save(model.state_dict(), output_model_file)  # save state & weights of model

    tokenizer.save_vocabulary(output_vocab_file)


if __name__ == '__main__':
    main()

# to kick off training call the main function