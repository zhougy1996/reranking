import torch
import pandas as pd
from transformers import BertTokenizer
from models import TextClassifier
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_path = '../models/pretrained/bert-base-uncased'

# Init tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_path)
# Init model
model = TextClassifier(bert_path, 2).to(DEVICE)
# Init dataset
dataset = pd.read_csv("../data/train_set.tsv", sep="\t")
# Init trainer
trainer = Trainer(model, tokenizer, dataset)


trainer.train(DEVICE, 1)
# Save trained model
torch.save(model, '../models/my_model.pkl')
