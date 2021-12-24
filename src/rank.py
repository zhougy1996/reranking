import os

import torch
import pandas as pd
import numpy
from transformers import BertTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_TEXT_LENGTH = 256

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, '..')
BERT_DIR = os.path.join(ROOT_DIR, 'models/pretrained/bert-base-uncased')
CURRENT_MODEL_FILE = os.path.join(ROOT_DIR, 'models/my_model.pkl')

# Init tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_DIR)

model = torch.load(CURRENT_MODEL_FILE, map_location=DEVICE)
model.eval()


def predict(query, passage):
    text = query + '[SEP]' + passage
    tokenized_text = tokenizer(text, max_length=MAX_TEXT_LENGTH, truncation=True, padding=True, return_tensors='pt').to(
        DEVICE)
    outputs = model(**tokenized_text)
    return outputs


def get_score(query, passage):
    score = predict(query, passage).data.numpy()[0][0]
    return score * 10


# 读取验证集（使用2019年的测试集作为验证集，43个验证查询）
# val_set
val_set = pd.read_csv(os.path.join(ROOT_DIR, 'data/msmarco-passagetest2019-43-top1000.tsv'), sep='\t', header=None)
val_set.columns = ['qid', 'pid', 'query', 'passage']
# print(val_set['qid'].value_counts().count())

# 读取测试集（2020年的测试集，54个测试查询）
# test_set
test_set = pd.read_csv(os.path.join(ROOT_DIR, 'data/msmarco-passagetest2020-54-top1000.tsv'), sep='\t', header=None)
test_set.columns = ['qid', 'pid', 'query', 'passage']
# print(test_set['qid'].value_counts().count())


# Do something here
criterion = torch.nn.CrossEntropyLoss()
q = "seraphina name meaning"
p = "Hebrew Meaning: The name Seraphina is a Hebrew baby name. In Hebrew the meaning of the name Seraphina is: " \
    "Fiery-winged. The name Seraphina comes from 'seraphim', who were the most powerful angels. "
# result = predict(q, p)
# loss_0 = criterion(result, torch.Tensor([[1, 0]])).data.numpy()
# loss_1 = criterion(result, torch.Tensor([[0, 1]])).data.numpy()
# print(loss_0)
# print(loss_1)
score = get_score(q, p)
print('score: {:.4f}'.format(score))
pass
