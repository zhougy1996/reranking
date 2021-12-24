import torch.nn
from transformers import BertModel, BertConfig
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, bert_path):
        super(TextClassifier, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.linear = torch.nn.Linear(self.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]
        linear_output = self.linear(out_pool)
        return self.sigmoid(linear_output)

