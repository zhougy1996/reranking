import torch
import pandas as pd
from transformers import BertTokenizer
from models import TextClassifier
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_path = '../models/pretrained/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_path)
model = TextClassifier(bert_path, 2).to(DEVICE)
dataset = pd.read_csv("../data/train_set.tsv", sep="\t")

trainer = Trainer(model, tokenizer, dataset)
trainer.train(DEVICE, 1)
torch.save(model, '../models/news_model.pkl')