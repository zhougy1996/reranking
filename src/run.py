import os

import torch
import pandas as pd
from transformers import BertTokenizer
from models import TextClassifier
from trainer import Trainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, '..')
BERT_DIR = os.path.join(ROOT_DIR, 'models/pretrained/bert-base-uncased')

# Init tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
# Init model
model = TextClassifier(BERT_DIR).to(DEVICE)
# Init dataset
dataset = pd.read_csv(os.path.join(ROOT_DIR, 'data/train_set.tsv'), sep='\t')
# Init trainer
trainer = Trainer(model, tokenizer, dataset)

# Train with device and epoch_num
trainer.train(DEVICE, 5)
# Save trained model
torch.save(model, os.path.join(ROOT_DIR, 'models/my_model.pkl'))
